"""
reachy_demo/memory.py — tiny persistent long-term memory so Reachy feels like it
actually remembers the people it meets.

Two layers work together in demo_tools7:

  1. Short-term: the in-process `history` list (this conversation). Already there.
  2. Long-term (this module): a handful of short, durable notes Reachy has learned
     across ALL past runs — names, interests, jobs, fun details — persisted in
     memory/reachy_memory.json and injected into the system prompt at the top of
     every session. New notes are pulled out by a cheap LLM call that runs in the
     BACKGROUND after each reply, so remembering never slows the conversation.

Because demo 7 has no per-visitor identity (every visitor is "the user"), the
memories are framed as general recollections ("a visitor named Mia loves Bitcoin")
and the prompt tells Reachy not to assume the current visitor is the same person —
it just lets Reachy charmingly bring up things it has heard before.
"""
import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).parent.parent
MEM_FILE = ROOT / "memory" / "reachy_memory.json"

MAX_MEMORIES = 60        # hard cap on stored notes (oldest dropped)
PROMPT_MEMORIES = 10     # how many most-recent notes to put in the system prompt

_lock = threading.Lock()


def load_memories() -> list[str]:
    """Return the stored memory notes (oldest→newest). [] if none / unreadable."""
    try:
        data = json.loads(MEM_FILE.read_text())
        return [m for m in data if isinstance(m, str)] if isinstance(data, list) else []
    except Exception:
        return []


def _save(mems: list[str]):
    # Write atomically (temp file + os.replace) so a concurrent, unlocked reader
    # in load_memories()/memory_block() — e.g. the main thread building the next
    # prompt while this runs in the background — never sees a half-written file
    # and silently loses all remembered facts for that turn.
    try:
        MEM_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = MEM_FILE.with_name(MEM_FILE.name + ".tmp")
        tmp.write_text(json.dumps(mems, ensure_ascii=False, indent=2))
        os.replace(tmp, MEM_FILE)
    except Exception:
        pass


def remember(texts) -> list[str]:
    """Add one or more short notes (str or list of str), case-insensitively
    de-duplicated and capped at MAX_MEMORIES. Returns the full updated list."""
    if isinstance(texts, str):
        texts = [texts]
    texts = [t.strip() for t in texts if t and t.strip()]
    with _lock:
        mems = load_memories()
        seen = {m.lower() for m in mems}
        for t in texts:
            if t.lower() not in seen:
                mems.append(t)
                seen.add(t.lower())
        mems = mems[-MAX_MEMORIES:]
        _save(mems)
        return mems


def memory_block(mems: list[str] | None = None) -> str:
    """Render the most-recent memories as a system-prompt block. '' if empty."""
    mems = load_memories() if mems is None else mems
    recent = mems[-PROMPT_MEMORIES:]
    if not recent:
        return ""
    lines = "\n".join(f"- {m}" for m in recent)
    return (
        "=== THINGS YOU REMEMBER FROM PAST CHATS ===\n"
        "Little things you've learned meeting people around NS. Bring one up "
        "naturally and warmly when it actually fits the conversation — it makes "
        "people feel remembered and delighted. Do NOT recite them or assume the "
        "current visitor is the same person.\n" + lines
    )


_EXTRACT_SYSTEM = (
    "You help a friendly robot remember people. From the LAST exchange, extract at "
    "most TWO short, durable facts genuinely worth remembering long-term: the "
    "visitor's name, where they're from, their job/interests, a promise made, or a "
    "fun personal detail. Write each as a tiny third-person note, e.g. 'A visitor "
    "named Tony loves Bitcoin.' or 'Someone is learning Mandarin at NS.' "
    "IGNORE greetings, small talk, questions about NS, and anything trivial or "
    "temporary. If nothing is worth remembering, reply with exactly: NONE. "
    "Output one note per line. No bullets, no quotes, no extra words."
)


def extract_memories(client, model: str, user_text: str, reply_text: str) -> list[str]:
    """Cheap LLM call: pull 0–2 durable notes from one exchange. [] on nothing/error.
    Meant to run in a background thread so it never blocks the conversation."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user",
                 "content": f"Visitor said: {user_text}\nReachy replied: {reply_text}"},
            ],
            max_tokens=60, temperature=0.2, stream=False,
        )
        out = (resp.choices[0].message.content or "").strip()
    except Exception:
        return []
    if not out or out.strip().upper().startswith("NONE"):
        return []
    facts = []
    for line in out.splitlines():
        f = line.strip().lstrip("-•*").strip().strip('"').strip()
        if f and f.upper() != "NONE":
            facts.append(f)
    return facts[:2]
