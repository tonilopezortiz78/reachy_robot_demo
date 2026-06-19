"""
reachy_demo/tts_edge.py — edge-tts synthesis utilities.

A single asyncio event loop runs in a background thread for the lifetime of
the process. All edge-tts calls are submitted to it via run_coroutine_threadsafe,
which reuses the underlying TLS connection instead of re-handshaking per sentence.

Provides:
  - ENGLISH_VOICE, CHINESE_VOICE — voice name constants
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

ENGLISH_VOICE = "en-US-AnaNeural"   # child voice — naturally high-pitched and cute
# YunyangNeural is Microsoft's newscast-style Mandarin voice.
# Newscast voices are trained with explicit tone precision and clear articulation —
# the best choice for a robot that needs to be understood in a noisy event space.
CHINESE_VOICE = "zh-CN-YunyangNeural"

# TTS tuning — snappier pace + louder output
# Rate: % offset from natural (positive = faster, negative = slower)
# Vol:  ffmpeg volume multiplier (1.0 = unity, 2.0 = +6 dB, 2.5 = +8 dB)
ENGLISH_RATE, ENGLISH_PITCH, ENGLISH_VOL = "+30%", "+8Hz", "2.5"
CHINESE_RATE, CHINESE_PITCH, CHINESE_VOL = "+5%",  "+8Hz", "2.5"

# ── Persistent event loop ─────────────────────────────────────────────────────

_tts_loop   = asyncio.new_event_loop()
_tts_thread = threading.Thread(target=_tts_loop.run_forever, daemon=True)
_tts_thread.start()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_chinese(text: str) -> bool:
    """True if more than 15% of characters are Chinese."""
    cjk = sum(1 for c in text if '一' <= c <= '鿿')
    return cjk > max(2, len(text) * 0.15)


async def _edge_synth_coro(text: str, mp3_path: str, voice: str, rate: str, pitch: str):
    tts = _edge_tts_mod.Communicate(text, voice=voice, rate=rate, pitch=pitch)
    await asyncio.wait_for(tts.save(mp3_path), timeout=30.0)

# ── Public API ────────────────────────────────────────────────────────────────

def synth_to_file(text: str) -> str:
    """
    Synthesise text via edge-tts, resample to 48kHz WAV, return temp path.
    Caller must delete the returned file.
    Language is auto-detected: Chinese → CHINESE_VOICE, else ENGLISH_VOICE.
    """
    mp3 = tempfile.mktemp(suffix=".mp3")
    out = tempfile.mktemp(suffix=".wav")
    if _is_chinese(text):
        voice, rate, pitch, vol = CHINESE_VOICE, CHINESE_RATE, CHINESE_PITCH, CHINESE_VOL
    else:
        voice, rate, pitch, vol = ENGLISH_VOICE, ENGLISH_RATE, ENGLISH_PITCH, ENGLISH_VOL
    snippet = text[:50].replace("\n", " ")
    try:
        t0 = time.time()
        future = asyncio.run_coroutine_threadsafe(
            _edge_synth_coro(text, mp3, voice, rate, pitch), _tts_loop
        )
        future.result(timeout=33.0)
        t_edge = time.time() - t0

        t1 = time.time()
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", mp3,
             "-af", f"aresample=resampler=swr:out_sample_rate=48000,volume={vol}",
             out],
            check=True,
        )
        t_ffmpeg = time.time() - t1

        print(f"  TTS  {t_edge:.2f}s edge  {t_ffmpeg:.2f}s ffmpeg  │ {snippet!r}", flush=True)
    finally:
        Path(mp3).unlink(missing_ok=True)
    return out
