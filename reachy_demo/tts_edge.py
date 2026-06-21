"""
reachy_demo/tts_edge.py — edge-tts synthesis utilities.

A single asyncio event loop runs in a background thread for the lifetime of
the process. All edge-tts calls are submitted to it via run_coroutine_threadsafe,
which reuses the underlying TLS connection instead of re-handshaking per sentence.

Provides:
  - VOICE — multilingual voice constant (same voice for all languages)
  - synth_to_file(text) → str  (temp WAV path; caller must delete)
  - play_wav_blocking(path) — re-exported from audio for convenience
"""

import asyncio
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import edge_tts as _edge_tts_mod  # import once at module level

from reachy_demo.audio import play_wav_blocking  # noqa: F401  (re-export)

# ── Voice constants ───────────────────────────────────────────────────────────

# Single multilingual voice — sounds identical regardless of language spoken.
# AvaMultilingual: Expressive, Caring, Pleasant, Friendly — ideal robot character.
VOICE = "en-US-AvaMultilingualNeural"

# TTS tuning — snappier pace + louder output + cute child pitch
# Rate:  % offset from natural (positive = faster, negative = slower)
# Pitch: +52Hz — a very high, childlike robot voice. AvaMultilingual is an adult
#        voice at 0Hz; this big lift is what makes Reachy sound like a little kid.
#        This is intentional. If it ever sounds too chipmunky, dial back toward +32Hz.
#        NEVER set to 0.
# Vol:   ffmpeg volume multiplier (1.0 = unity, 2.0 = +6 dB, 2.5 = +8 dB)
RATE, PITCH, VOL = "+25%", "+48Hz", "2.5"

# ── Persistent event loop ─────────────────────────────────────────────────────

_tts_loop   = asyncio.new_event_loop()
_tts_thread = threading.Thread(target=_tts_loop.run_forever, daemon=True)
_tts_thread.start()

# ── Helpers ───────────────────────────────────────────────────────────────────

async def _edge_synth_coro(text: str, mp3_path: str, voice: str, rate: str, pitch: str):
    tts = _edge_tts_mod.Communicate(text, voice=voice, rate=rate, pitch=pitch)
    await asyncio.wait_for(tts.save(mp3_path), timeout=30.0)

# ── Public API ────────────────────────────────────────────────────────────────

def synth_to_file(text: str) -> str:
    """
    Synthesise text via edge-tts, resample to 48kHz WAV, return temp path.
    Caller must delete the returned file.
    Uses a single multilingual voice — language is passed through as-is.
    """
    mp3 = tempfile.mktemp(suffix=".mp3")
    out = tempfile.mktemp(suffix=".wav")
    snippet = text[:50].replace("\n", " ")
    try:
        t0 = time.time()
        future = asyncio.run_coroutine_threadsafe(
            _edge_synth_coro(text, mp3, VOICE, RATE, PITCH), _tts_loop
        )
        future.result(timeout=33.0)
        t_edge = time.time() - t0

        t1 = time.time()
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", mp3,
             "-af", f"aresample=resampler=swr:out_sample_rate=48000,volume={VOL}",
             out],
            check=True,
        )
        t_ffmpeg = time.time() - t1

        print(f"  TTS  {t_edge:.2f}s edge  {t_ffmpeg:.2f}s ffmpeg  │ {snippet!r}", flush=True)
    finally:
        Path(mp3).unlink(missing_ok=True)
    return out
