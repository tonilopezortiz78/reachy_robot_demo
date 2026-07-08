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
import queue as _queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import edge_tts as _edge_tts_mod  # import once at module level

from reachy_demo.audio import SPEAKER, play_wav_blocking  # noqa: F401  (re-export)

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
RATE, PITCH, VOL = "+20%", "+48Hz", "2.5"

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
    except BaseException:
        Path(out).unlink(missing_ok=True)   # don't leak the WAV on failure
        raise
    finally:
        Path(mp3).unlink(missing_ok=True)
    return out


# ── Streaming playback (near-instant time-to-first-audio) ──────────────────────

def stream_to_speaker(text: str, stop_check=None, on_first_audio=None,
                      speaker: str = SPEAKER) -> bool:
    """
    Stream edge-tts audio STRAIGHT to the robot speaker as chunks arrive, instead
    of synthesising the whole sentence to a file first. Audio starts ~0.4s after
    the call (vs ~3.6s for synth-then-play) — this is what makes demo 6 feel
    near-instant.

    How: edge-tts `.stream()` yields MP3 chunks from Microsoft as they're
    generated; we pipe them into `ffmpeg -i pipe:0 -f alsa <speaker>`, which
    decodes and plays incrementally. The persistent event loop produces chunks
    into a thread-safe queue; this (calling) thread feeds ffmpeg's stdin.

    stop_check():    optional callable -> bool. Polled continuously; if it ever
                     returns True, playback aborts immediately (barge-in).
    on_first_audio(): optional callable fired once, when the first audio chunk is
                     about to play (caller can switch to SPEAKING / kill ticks).

    Returns True if it played to completion, False if aborted by stop_check or
    if synthesis/playback failed (edge-tts error, ffmpeg died mid-stream).
    """
    snippet = text[:50].replace("\n", " ")
    q: "_queue.Queue" = _queue.Queue()
    _DONE = object()

    async def _produce():
        try:
            c = _edge_tts_mod.Communicate(text, voice=VOICE, rate=RATE, pitch=PITCH)
            async for chunk in c.stream():
                if chunk["type"] == "audio":
                    q.put(chunk["data"])
        except Exception as e:                       # network hiccup, etc.
            q.put(("ERR", e))
        finally:
            q.put(_DONE)

    fut = asyncio.run_coroutine_threadsafe(_produce(), _tts_loop)

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "mp3", "-i", "pipe:0",
         "-af", f"volume={VOL}",
         "-f", "alsa", speaker],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    t0 = time.time()
    first = True
    aborted = False
    failed = False
    err = None
    try:
        while True:
            if stop_check and stop_check():
                aborted = True
                break
            try:
                item = q.get(timeout=0.05)
            except _queue.Empty:
                continue
            if item is _DONE:
                break
            if isinstance(item, tuple) and item and item[0] == "ERR":
                err = item[1]
                failed = True
                break
            if first:
                first = False
                if on_first_audio:
                    on_first_audio()
                print(f"  TTS  {time.time()-t0:.2f}s to first audio (stream) │ {snippet!r}",
                      flush=True)
            try:
                ff.stdin.write(item)
            except (BrokenPipeError, ValueError):
                failed = True           # ffmpeg died mid-sentence
                break
    finally:
        try:
            ff.stdin.close()
        except Exception:
            pass
        if aborted:
            fut.cancel()
            ff.terminate()
            try:
                ff.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                ff.kill(); ff.wait()
        else:
            # Drain: ffmpeg still has buffered audio to play. Keep checking
            # barge-in so the user can interrupt the tail of a sentence too.
            while ff.poll() is None:
                if stop_check and stop_check():
                    aborted = True
                    ff.terminate()
                    try:
                        ff.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        ff.kill(); ff.wait()
                    break
                time.sleep(0.03)
    if err is not None:
        print(f"  TTS  stream error: {err}", flush=True)
    return not aborted and not failed
