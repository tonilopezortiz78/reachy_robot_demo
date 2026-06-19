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
from pathlib import Path

import edge_tts as _edge_tts_mod  # import once at module level

from reachy_demo.audio import play_wav_blocking  # noqa: F401  (re-export)

# ── Voice constants ───────────────────────────────────────────────────────────

ENGLISH_VOICE  = "en-US-AnaNeural"    # child voice — naturally high-pitched and cute
ENGLISH_STYLE  = "cheerful"            # SSML express-as style (AnaNeural supports cheerful/excited/sad/empathetic)
CHINESE_VOICE  = "zh-CN-YunyangNeural"

# ── Persistent event loop ─────────────────────────────────────────────────────

_tts_loop   = asyncio.new_event_loop()
_tts_thread = threading.Thread(target=_tts_loop.run_forever, daemon=True)
_tts_thread.start()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_chinese(text: str) -> bool:
    """True if more than 15% of characters are Chinese."""
    cjk = sum(1 for c in text if '一' <= c <= '鿿')
    return cjk > max(2, len(text) * 0.15)


def _ssml(text: str, voice: str, rate: str, pitch: str, style: str) -> str:
    return (
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
        "xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='en-US'>"
        f"<voice name='{voice}'><prosody rate='{rate}' pitch='{pitch}'>"
        f"<mstts:express-as style='{style}' styledegree='1.5'>{text}</mstts:express-as>"
        "</prosody></voice></speak>"
    )


async def _edge_synth_coro(text: str, mp3_path: str, voice: str, rate: str,
                            pitch: str, style: str | None = None):
    content = _ssml(text, voice, rate, pitch, style) if style else text
    tts = _edge_tts_mod.Communicate(content) if style else \
          _edge_tts_mod.Communicate(text, voice=voice, rate=rate, pitch=pitch)
    await asyncio.wait_for(tts.save(mp3_path), timeout=10.0)

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
        voice, rate, pitch, style, vol = CHINESE_VOICE, "-18%", "+0Hz", None, "2.2"
    else:
        voice, rate, pitch, style, vol = ENGLISH_VOICE, "+20%", "+8Hz", ENGLISH_STYLE, "2.0"
    try:
        future = asyncio.run_coroutine_threadsafe(
            _edge_synth_coro(text, mp3, voice, rate, pitch, style), _tts_loop
        )
        future.result(timeout=12.0)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", mp3,
             "-af", f"aresample=resampler=swr:out_sample_rate=48000,volume={vol}",
             out],
            check=True,
        )
    finally:
        Path(mp3).unlink(missing_ok=True)
    return out
