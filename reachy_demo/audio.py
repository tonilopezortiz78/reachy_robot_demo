"""
reachy_demo/audio.py — Audio utilities for Reachy talking demos.

Covers:
  - Hardware constants (SPEAKER, MIC)
  - Tone generators (_beep, blip, chirp)
  - Named sound effects (listening_ping, speaking_chime, error_chime,
    your_turn_chime, boot_beeps, thinking_blips)
  - play_wav_blocking() — play a WAV file on the robot speaker, blocking
  - VAD constants and record_utterance() — full VAD capture loop
  - pcm_to_wav_bytes() — wrap raw PCM in a WAV container
"""

import io
import subprocess
import time
import wave

import numpy as np
import torch
from silero_vad import VADIterator

# ── Hardware constants ────────────────────────────────────────────────────────

SPEAKER = "plughw:CARD=Audio,DEV=0"
MIC     = "alsa_input.pci-0000_00_1f.3.analog-stereo"   # laptop mic via PipeWire

# ── VAD constants ─────────────────────────────────────────────────────────────

MIC_RATE       = 16000
VAD_CHUNK      = 512          # 32 ms per chunk — Silero's native size
SPEECH_THRESH  = 0.45         # VAD confidence threshold
SILENCE_END_MS = 1400         # ms of silence → end of utterance (VADIterator internal)
TAIL_FRAMES    = 18           # extra chunks (~576ms) collected after "end" detected
MIN_SPEECH_S   = 0.4          # ignore very short blips (< 400 ms)
MAX_RECORD_S   = 15.0         # safety cap

# ── Tone generators ───────────────────────────────────────────────────────────

def _beep(expr, dur, vol=0.5, block=True):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", f"aevalsrc={expr}*{vol}:c=mono:s=22050",
           "-t", str(dur), "-f", "alsa", SPEAKER]
    if block:
        subprocess.run(cmd, check=False)
    else:
        subprocess.Popen(cmd)


def blip(freq, dur=0.07, vol=0.4, block=True):
    _beep(f"sin(2*PI*{freq}*t)*exp(-t*8)", dur, vol, block)


def chirp(f0, f1, dur, vol=0.45, block=True):
    _beep(f"sin(2*PI*({f0}*t+({f1}-{f0})*t*t/(2*{dur})))", dur, vol, block)

# ── Named sound effects ───────────────────────────────────────────────────────

def boot_beeps():
    for f, d in [(300, 0.07), (500, 0.06), (750, 0.07), (1100, 0.06), (1600, 0.05)]:
        blip(f, d, 0.38, block=True)
        time.sleep(0.03)
    time.sleep(0.04)
    blip(2000, 0.06, 0.42, block=True)


def listening_ping():
    """Soft tick while waiting for someone to speak."""
    chirp(500, 1200, 0.09, vol=0.35, block=False)


def your_turn_chime():
    """Clear 3-note rising signal: robot finished, your turn to speak."""
    for f in [600, 900, 1400]:
        blip(f, 0.07, 0.55, block=True)
        time.sleep(0.05)


def thinking_blips():
    for f in [700, 550, 400]:
        blip(f, 0.05, 0.25, block=True)
        time.sleep(0.04)


def speaking_chime():
    blip(900, 0.06, 0.28, block=True)


def error_chime():
    chirp(400, 180, 0.25, vol=0.30, block=True)

# ── WAV playback ──────────────────────────────────────────────────────────────

def play_wav_blocking(path: str):
    """Play a WAV file on the robot speaker and block until done."""
    proc = subprocess.Popen(
        ["aplay", "-D", SPEAKER, "-q", path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    proc.wait()
    time.sleep(0.06)

# ── VAD capture ───────────────────────────────────────────────────────────────

def record_utterance(vad_model, ping=None) -> bytes | None:
    """
    Capture mic via arecord, feed to Silero VAD.
    VADIterator internally waits SILENCE_END_MS of silence before signalling "end".
    We then collect TAIL_FRAMES extra chunks so the tail of the word isn't cut.
    Returns raw PCM bytes (int16, 16kHz, mono), or None if too short.

    ping: optional callable to emit a start-of-listening sound.
          Defaults to listening_ping() (non-blocking chirp).
          Pass a custom callable to override (e.g. a blocking version).
    """
    if ping is None:
        ping = listening_ping

    vad_iter = VADIterator(vad_model, sampling_rate=MIC_RATE,
                           threshold=SPEECH_THRESH,
                           min_silence_duration_ms=SILENCE_END_MS)

    arecord = subprocess.Popen(
        ["pacat", "--record", "--raw",
         f"--device={MIC}",
         f"--rate={MIC_RATE}", "--channels=1", "--format=s16le"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    print("  Listening...", end="", flush=True)
    ping()

    speech_buf  = []
    in_speech   = False
    ended       = False
    tail_count  = 0
    max_frames  = int(MAX_RECORD_S * MIC_RATE / VAD_CHUNK)
    total       = 0

    try:
        while total < max_frames:
            raw = arecord.stdout.read(VAD_CHUNK * 2)
            if not raw or len(raw) < VAD_CHUNK * 2:
                break

            audio_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            result = vad_iter(torch.from_numpy(audio_f32))

            if result and "start" in result and not in_speech:
                in_speech = True
                print(" ●", end="", flush=True)

            if in_speech:
                speech_buf.append(raw)

            if result and "end" in result and in_speech and not ended:
                ended = True
                print(" ◼", end="", flush=True)

            if ended:
                tail_count += 1
                if tail_count >= TAIL_FRAMES:
                    break

            total += 1

    finally:
        arecord.terminate()
        arecord.wait()

    print()

    min_frames = int(MIN_SPEECH_S * MIC_RATE / VAD_CHUNK)
    if len(speech_buf) < min_frames:
        return None
    return b"".join(speech_buf)


def pcm_to_wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw int16/16kHz/mono PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(MIC_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()
