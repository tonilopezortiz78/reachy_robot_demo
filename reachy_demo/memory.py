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
import re
import threading
from pathlib import Path

ROOT = Path(__file__).parent.parent
MEM_FILE = ROOT / "memory" / "reachy_memory.json"
PEOPLE_DIR = ROOT / "cache" / "people"

MAX_MEMORIES = 60        # hard cap on stored notes (oldest dropped)
PROMPT_MEMORIES = 10     # how many most-recent notes to put in the system prompt

MAX_PERSON_FACTS = 25    # hard cap on stored facts per person (oldest dropped)
PROMPT_PERSON_FACTS = 8  # how many most-recent facts to put in the system prompt

_lock = threading.Lock()
_person_lock = threading.Lock()


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


# --- Per-person memory ------------------------------------------------------
# Additional, separate from the global pool above: durable facts tied to a
# specific recognized person (e.g. "Tony is from Spain"), so the robot can
# recall them the next time it recognizes that same face. Stored one JSON
# file per person under cache/people/ so the existing global memory file and
# its callers are completely unaffected.

def _person_slug(name: str) -> str:
    """Filesystem-safe slug for a person's name. '' for empty/unknown/visitor."""
    if not name:
        return ""
    name = name.strip().lower()
    if not name or name == "visitor":
        return ""
    slug = re.sub(r"\s+", "_", name)
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    return slug


def _person_path(name: str) -> Path:
    """Path to the per-person facts file for `name` (may not exist yet)."""
    slug = _person_slug(name)
    return PEOPLE_DIR / f"{slug}.json"


def load_person_facts(name: str) -> list[str]:
    """Return the stored facts for `name` (oldest→newest). [] if none/unknown/visitor."""
    slug = _person_slug(name)
    if not slug:
        return []
    try:
        data = json.loads(_person_path(name).read_text())
        facts = data.get("facts") if isinstance(data, dict) else None
        return [f for f in facts if isinstance(f, str)] if isinstance(facts, list) else []
    except Exception:
        return []


def remember_person(name: str, facts) -> None:
    """Add one or more short facts about `name` (str or list of str),
    case-insensitively de-duplicated and capped at MAX_PERSON_FACTS. No-op if
    name is empty/"visitor". Never raises."""
    slug = _person_slug(name)
    if not slug:
        return
    if isinstance(facts, str):
        facts = [facts]
    try:
        facts = [f.strip() for f in facts if f and f.strip()]
    except Exception:
        return
    if not facts:
        return
    try:
        with _person_lock:
            existing = load_person_facts(name)
            seen = {f.lower() for f in existing}
            for f in facts:
                if f.lower() not in seen:
                    existing.append(f)
                    seen.add(f.lower())
            existing = existing[-MAX_PERSON_FACTS:]
            path = _person_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(
                {"name": name.strip(), "facts": existing},
                ensure_ascii=False, indent=2,
            ))
            os.replace(tmp, path)
    except Exception:
        pass


def set_person_facts(name: str, facts) -> bool:
    """Replace ALL stored facts for `name` with `facts` (list of str), for
    operator add/edit/delete from the dashboard. Trims/de-dupes/caps like
    remember_person but overwrites instead of appending. Returns True on write.
    Never raises."""
    slug = _person_slug(name)
    if not slug:
        return False
    if isinstance(facts, str):
        facts = [facts]
    try:
        clean, seen = [], set()
        for f in facts:
            if not isinstance(f, str):
                continue
            f = f.strip()
            if f and f.lower() not in seen:
                clean.append(f)
                seen.add(f.lower())
        clean = clean[-MAX_PERSON_FACTS:]
        with _person_lock:
            path = _person_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(
                {"name": name.strip(), "facts": clean},
                ensure_ascii=False, indent=2,
            ))
            os.replace(tmp, path)
        return True
    except Exception:
        return False


def delete_person_memory(name: str) -> bool:
    """Delete the per-person facts file for `name` (used when an enrolled person
    is removed from the dashboard). Returns True if a file was removed. Never raises."""
    slug = _person_slug(name)
    if not slug:
        return False
    try:
        with _person_lock:
            p = _person_path(name)
            if p.exists():
                p.unlink()
                return True
    except Exception:
        pass
    return False


def person_summary_block(name: str) -> str:
    """Render the most-recent facts about `name` as a compact system-prompt
    block. '' if no facts / unknown / visitor."""
    facts = load_person_facts(name)
    recent = facts[-PROMPT_PERSON_FACTS:]
    if not recent:
        return ""
    lines = "\n".join(f"- {f}" for f in recent)
    return f"What you remember about {name.strip()}:\n{lines}"


def known_people() -> list[str]:
    """Names of all people with a stored profile file. [] if none / unreadable."""
    try:
        PEOPLE_DIR.mkdir(parents=True, exist_ok=True)
        names = []
        for path in sorted(PEOPLE_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                name = data.get("name") if isinstance(data, dict) else None
                names.append(name if isinstance(name, str) and name else path.stem)
            except Exception:
                continue
        return names
    except Exception:
        return []


def rename_person_facts(old_name: str, new_name: str) -> bool:
    """Move the per-person facts file old→new slug, updating the stored "name".
    If new already has facts (a merge), old facts are folded in de-duplicated so
    a rename never destroys data. False if old has no file or a slug is invalid;
    True (no-op) if slugs match. Never raises."""
    old_slug = _person_slug(old_name)
    new_slug = _person_slug(new_name)
    if not old_slug or not new_slug:
        return False
    if old_slug == new_slug:
        return True
    old_path = PEOPLE_DIR / f"{old_slug}.json"
    with _person_lock:
        if not old_path.exists():
            return False
        try:
            data = json.loads(old_path.read_text())
            old_facts = data.get("facts") if isinstance(data, dict) else None
            old_facts = [f for f in old_facts if isinstance(f, str)] if isinstance(old_facts, list) else []
        except Exception:
            old_facts = []

        new_path = PEOPLE_DIR / f"{new_slug}.json"
        try:
            data = json.loads(new_path.read_text())
            existing = data.get("facts") if isinstance(data, dict) else None
            existing = [f for f in existing if isinstance(f, str)] if isinstance(existing, list) else []
        except Exception:
            existing = []

        seen = {f.lower() for f in existing}
        merged = list(existing)
        for f in old_facts:
            if f.lower() not in seen:
                merged.append(f)
                seen.add(f.lower())
        merged = merged[-MAX_PERSON_FACTS:]

        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = new_path.with_name(new_path.name + ".tmp")
            tmp.write_text(json.dumps(
                {"name": new_name.strip(), "facts": merged},
                ensure_ascii=False, indent=2,
            ))
            os.replace(tmp, new_path)
        except Exception:
            return False
        try:
            old_path.unlink(missing_ok=True)
        except Exception:
            pass
        return True
