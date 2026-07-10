"""reachy_demo/phrases.py — pre-rendered quick phrases for the dashboard.

The operator's "Quick phrases" buttons should fire INSTANTLY. A live edge-tts
call costs a ~0.4-0.5 s network round-trip; instead we synthesise each phrase
once to a cached WAV (unity gain) and replay it locally through
tts_edge.play_wav_file(), which applies the current volume + robot/projector
routing at play time. First run pays the synth cost once; the WAVs persist on
disk in cache/phrases/ across restarts.

Single source of truth for the phrase list — the dashboard buttons are rendered
from QUICK_PHRASES (injected into the page) and the demo matches spoken text
against the same list to decide whether a cached WAV exists.
"""
from __future__ import annotations

import hashlib
import shutil
import threading
from pathlib import Path

# (button label, spoken text). Labels carry an emoji so the grid reads fast.
QUICK_PHRASES: list[tuple[str, str]] = [
    ("👋 Welcome",   "Welcome to Network School, everyone!"),
    ("🧒 Hi kids",   "Hi kids! I'm Reachy, your little robot friend!"),
    ("🤖 About me",  "I'm a Reachy Mini robot. I can see you, hear you, and talk back!"),
    ("💃 Dance?",    "Do you want to see me dance? Just ask me!"),
    ("🌍 Languages", "Try talking to me in any language — I understand lots of them!"),
    ("😄 Joke",      "Why did the robot go on vacation? To recharge its batteries!"),
    ("👏 Wow",       "Wow, that is so cool!"),
    ("🙌 Bye",       "Thanks for visiting! Onward and upward!"),
]

_CACHE = Path(__file__).resolve().parent.parent / "cache" / "phrases"


def _key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def cached_wav(text: str) -> str | None:
    """Return the cached WAV path for `text` if it has been pre-rendered, else None."""
    p = _CACHE / f"{_key(text)}.wav"
    return str(p) if p.exists() else None


def prerender(log=None) -> None:
    """Synthesise every phrase to a cached unity-gain WAV (idempotent). Safe to
    call at startup — skips phrases already on disk, swallows synth failures
    (flaky wifi just means that phrase falls back to live TTS at click time)."""
    # Imported lazily so importing this module never drags in the TTS event loop.
    from reachy_demo.tts_edge import synth_to_file

    _CACHE.mkdir(parents=True, exist_ok=True)
    made = 0
    for _label, text in QUICK_PHRASES:
        dest = _CACHE / f"{_key(text)}.wav"
        if dest.exists():
            continue
        try:
            tmp = synth_to_file(text, boost=False)   # raw voice; volume applied at play
            # Stage into the cache dir then atomically rename, so cached_wav()
            # can never hand a half-written WAV to a button click mid-prerender.
            staged = dest.with_suffix(".tmp")
            shutil.move(tmp, staged)
            staged.replace(dest)
            made += 1
        except Exception:
            continue
    if log is not None:
        cached = sum(1 for _l, t in QUICK_PHRASES if (_CACHE / f"{_key(t)}.wav").exists())
        log.event(f"  Quick phrases: {cached}/{len(QUICK_PHRASES)} cached "
                  f"({made} rendered this run)")


def prerender_async(log=None) -> None:
    """Fire-and-forget prerender on a daemon thread so startup never blocks on it."""
    threading.Thread(target=prerender, args=(log,), daemon=True).start()
