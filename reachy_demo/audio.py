"""
reachy_demo/audio.py — Audio utilities for Reachy talking demos.

Covers:
  - Hardware constants (SPEAKER, MIC)
  - Tone generators (_beep, blip, chirp)
  - Named sound effects (listening_ping, speaking_chime, error_chime,
    your_turn_chime, boot_beeps, thinking_blips, start_thinking_ticks)
  - play_wav_blocking() — play a WAV file on the robot speaker, blocking
  - VAD constants and record_utterance() — full VAD capture loop
  - pcm_to_wav_bytes() — wrap raw PCM in a WAV container
"""

import io
import subprocess
import threading
import time
import wave

import numpy as np
import torch
from silero_vad import VADIterator

# ── Hardware constants ────────────────────────────────────────────────────────

SPEAKER = "plughw:CARD=Audio,DEV=0"

# ── Microphone selection ──────────────────────────────────────────────────────
# IMPORTANT: this machine has MULTIPLE microphones. We MUST use the robot's own
# mic, not the laptop's — the laptop mic captures room noise instead of the
# visitor, which makes Whisper hallucinate words and mis-detect the language
# (the classic "I spoke Japanese, it replied Spanish" failure).
#
# Empirically verified on this hardware (pactl + RMS level check):
#   - alsa_input ...Reachy_Mini_Audio...   → REAL working voice mic, native 16kHz,
#                                            strong signal (RMS ~880). USE THIS.
#   - alsa_input ...Reachy_Mini_Camera...  → flatlined / silent (RMS ~2) on this
#                                            unit, despite the camera "having" a mic.
#   - alsa_input ...pci... (laptop)        → room noise, wrong device.
# (Note: this contradicts older notes that said the Audio device is playback-only.
#  Trust the measurement — the Pollen Audio device carries the working mic here.)
#
# Source names contain a per-unit serial, so we auto-detect at import. To inspect:
#     pactl list short sources
_LAPTOP_MIC_FALLBACK = "alsa_input.pci-0000_00_1f.3.analog-stereo"

# Preference order: robot Audio mic (works) → robot Camera mic → laptop fallback.
_MIC_PREFERENCE = ("Reachy_Mini_Audio", "Reachy_Mini_Camera")


def _detect_robot_mic(default: str) -> str:
    """Return the PipeWire source name of the robot's working mic, else `default`."""
    try:
        out = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        sources = [ln.split("\t")[1] for ln in out.splitlines()
                   if len(ln.split("\t")) >= 2 and ln.split("\t")[1].startswith("alsa_input")
                   and ".monitor" not in ln.split("\t")[1]]
        for want in _MIC_PREFERENCE:
            for name in sources:
                if want in name:
                    return name
    except Exception:
        pass
    return default


MIC = _detect_robot_mic(_LAPTOP_MIC_FALLBACK)   # robot voice mic, auto-detected

# ── VAD constants ─────────────────────────────────────────────────────────────

MIC_RATE       = 16000
VAD_CHUNK      = 512          # 32 ms per chunk — Silero's native size
SPEECH_THRESH  = 0.45         # VAD confidence threshold
SILENCE_END_MS = 800          # ms of silence → end of utterance (was 1400 — felt sluggish)
TAIL_FRAMES    = 10           # extra chunks (~320ms) collected after "end" detected
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


def chirp_nb(f0, f1, dur, vol=0.30):
    """Non-blocking chirp (frequency sweep). Fire-and-forget for tick loops."""
    _beep(f"sin(2*PI*({f0}*t+({f1}-{f0})*t*t/(2*{dur})))", dur, vol, block=False)

# ── Named sound effects ───────────────────────────────────────────────────────

def boot_beeps():
    """R2-D2-style startup — ascending sequence + happy double-blip."""
    for f, d, v in [(280, 0.06, 0.35), (420, 0.06, 0.37), (600, 0.07, 0.40),
                    (850, 0.07, 0.42), (1100, 0.06, 0.44), (1500, 0.07, 0.46)]:
        blip(f, d, v, block=True)
        time.sleep(0.025)
    time.sleep(0.06)
    chirp(800, 1800, 0.12, vol=0.50, block=True)
    time.sleep(0.03)
    blip(2200, 0.05, 0.45, block=True)
    time.sleep(0.02)
    blip(2200, 0.05, 0.45, block=True)


def listening_ping():
    """Soft rising tick — robot is awake and listening."""
    chirp(600, 1400, 0.08, vol=0.38, block=False)


def your_turn_chime():
    """4-note rising fanfare: robot finished, your turn."""
    for f, d in [(550, 0.06), (750, 0.06), (1050, 0.07), (1500, 0.09)]:
        blip(f, d, 0.52, block=True)
        time.sleep(0.04)
    time.sleep(0.02)
    chirp(1500, 900, 0.07, vol=0.30, block=True)


def thinking_blips():
    """4 descending blips + 1 ascending — 'computing' feel."""
    for f in [900, 720, 560, 420]:
        blip(f, 0.05, 0.22, block=True)
        time.sleep(0.035)
    time.sleep(0.02)
    blip(650, 0.06, 0.28, block=True)


def start_thinking_ticks(stop_event: threading.Event) -> threading.Thread:
    """
    Spawn a background thread that emits a slow sci-fi 'thinking scan' in a
    loop until `stop_event` is set. Use while waiting for STT/LLM so the user
    sees the robot is alive but hasn't fallen asleep.

    Call site:
        stop = threading.Event()
        start_thinking_ticks(stop)
        ... do STT + LLM + TTS prep ...
        stop.set()   # kills the loop immediately, before audio plays

    The scan is a rising chirp (low → high), followed by a long gap. Sounds
    like a robot actively reasoning, not a clock. Sleeps are broken into
    50ms slices so stop.set() takes effect within ~50ms.
    """
    def _run():
        # pattern: (kind, f0, f1, dur, vol, gap)
        #   chirp  : sweep from f0 → f1 Hz over `dur` seconds at `vol` (0-1)
        #   gap    : pause for `gap` seconds
        # Slow sci-fi scan + longer gap feels like the robot is "scanning"
        # its memory for an answer, not ticking like a clock.
        pattern = [
            ("chirp", 380, 1100, 0.55, 0.22, 0.0),
            ("gap",     0,    0, 0.00, 0.00, 1.30),
            ("chirp", 460, 1300, 0.40, 0.20, 0.0),
            ("gap",     0,    0, 0.00, 0.00, 1.50),
        ]
        while not stop_event.is_set():
            for kind, f0, f1, dur, vol, gap in pattern:
                if stop_event.is_set():
                    return
                if kind == "chirp":
                    chirp_nb(f0, f1, dur, vol)
                # break the pause into 50ms slices so stop is responsive
                end = time.time() + (dur if kind == "chirp" else gap)
                while time.time() < end:
                    if stop_event.is_set():
                        return
                    time.sleep(min(0.05, end - time.time()))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def speaking_chime():
    """3-note happy little 'I have something to say!' sequence."""
    for f, d in [(700, 0.05), (1000, 0.05), (1400, 0.07)]:
        blip(f, d, 0.38, block=True)
        time.sleep(0.03)


def error_chime():
    """Sad descending wobble."""
    chirp(500, 220, 0.28, vol=0.32, block=True)
    time.sleep(0.04)
    blip(200, 0.10, 0.25, block=True)


# ── Conversation state cues (clear & distinct) ────────────────────────────────
# Two unmistakable signals so the user always knows whose turn it is:
#   ready_cue()    — bright RISING two-note "beep-BOOP↑" = "I'm ready, your turn!"
#   thinking_cue() — soft  FALLING  "boo-doo↓" pulse     = "let me think..."
# Rising = your turn to talk; falling = I'm busy thinking. Easy to tell apart.

def ready_cue():
    """'I'm ready — talk to me now!' Bright rising two-tone. Blocking (no rush here)."""
    blip(784, 0.10, 0.48, block=True)    # G5
    time.sleep(0.03)
    blip(1245, 0.16, 0.55, block=True)   # D#6 — rising = open invitation to speak


def thinking_cue():
    """'Let me think...' Gentle descending sweep. NON-blocking so STT starts instantly."""
    chirp(820, 430, 0.32, vol=0.34, block=False)

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
