"""
reachy_demo/session_log.py — full conversation + API recorder for debugging.

Creates one sequentially-numbered folder per run under logs/ and records
EVERYTHING needed to diagnose a bad interaction after the fact:

  logs/1/   logs/2/   logs/3/  ...   (auto-incremented each run)
    console.log         — human-readable timeline of every event
    transcript.jsonl    — one JSON object per turn (STT, language, LLM payload, reply, timings)
    audio/turn_NNN.wav  — the exact audio Whisper heard each turn (replayable)
    audio/reply_NNN_S.wav — each TTS sentence Reachy spoke back (replayable)

Usage:
    log = SessionLogger(ROOT, "demo_tools7")
    log.event("Listening...")
    wav_path = log.save_audio(pcm)               # save what the mic captured
    log.turn(                                    # record the full turn
        audio=wav_path,
        whisper_lang="Japanese", whisper_text="こんにちは",
        final_lang="Japanese", directive="...",
        llm_messages=[...], reply="...", timings={"stt": 0.4, "llm": 0.6})

Everything is also echoed to stdout so the console is verbose too.
"""

import json
import shutil
import wave
from datetime import datetime
from pathlib import Path


def _next_interaction_dir(data_root: Path) -> Path:
    """Return logs/<N>/ where N is the next free sequential integer (1, 2, 3...)."""
    data_root.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name) for p in data_root.iterdir()
                if p.is_dir() and p.name.isdigit()]
    n = max(existing, default=0) + 1
    return data_root / str(n)


def _prune_old_sessions(data_root: Path, keep: int = 3) -> None:
    """Delete all but the <keep> most-recent numbered session directories."""
    dirs = sorted(
        [p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
        reverse=True,
    )
    for old in dirs[keep:]:
        shutil.rmtree(old, ignore_errors=True)


class SessionLogger:
    def __init__(self, root, demo_name: str, keep_sessions: int = 3):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_root = Path(root) / "logs"
        self.dir = _next_interaction_dir(data_root)
        self.number = self.dir.name
        self.audio_dir = self.dir / "audio"
        # Create the new session dir BEFORE pruning, then keep the N most recent.
        # (Pruning before creation would always leave N old + 1 new = N+1 dirs.)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        _prune_old_sessions(data_root, keep=keep_sessions)
        self.jsonl_path = self.dir / "transcript.jsonl"
        self.console_path = self.dir / "console.log"
        self._turn = 0
        self.event(f"=== INTERACTION {self.number}: {demo_name} @ {ts} ===")
        self.event(f"Recording to: {self.dir}")

    # ── Human-readable timeline ──────────────────────────────────────────────
    def event(self, msg: str, echo: bool = True):
        """Append a timestamped line to console.log (and optionally print it)."""
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{stamp}] {msg}"
        if echo:
            print(line, flush=True)
        with open(self.console_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ── Save the raw mic audio for a turn ────────────────────────────────────
    def save_audio(self, pcm: bytes, rate: int = 16000) -> str:
        """Write the captured PCM to audio/turn_NNN.wav and return the path."""
        self._turn += 1
        path = self.audio_dir / f"turn_{self._turn:03d}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        return str(path)

    # ── Save a played TTS reply wav so the session is fully replayable ────────
    def save_reply_wav(self, src_path: str, seg: int = 1) -> str | None:
        """Copy a played TTS wav into audio/reply_NNN_S.wav (NNN = current turn,
        S = segment index). Lets you replay BOTH sides of the conversation —
        what the mic heard (turn_NNN.wav) and what Reachy said (reply_NNN_S.wav).
        Best-effort: returns the new path, or None if the copy failed."""
        try:
            dst = self.audio_dir / f"reply_{self._turn:03d}_{seg}.wav"
            shutil.copyfile(src_path, dst)
            return str(dst)
        except Exception:
            return None

    # ── Structured per-turn record ───────────────────────────────────────────
    def turn(self, **fields):
        """Append one JSON record (all fields) to transcript.jsonl."""
        fields = {"turn": self._turn, "ts": datetime.now().isoformat(), **fields}
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fields, ensure_ascii=False) + "\n")

    def error(self, where: str, exc: Exception):
        self.event(f"!! ERROR in {where}: {type(exc).__name__}: {exc}")
        self.turn(error=f"{where}: {type(exc).__name__}: {exc}")
