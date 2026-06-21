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
# Source names contain a per-unit serial, so we auto-detect at import. We match
# on the SUBSTRING "Reachy_Mini_Audio" (not the full name), so a different unit's
# serial — or a PipeWire profile/port change — never breaks detection. To inspect:
#     pactl list short sources
_LAPTOP_MIC_FALLBACK = "alsa_input.pci-0000_00_1f.3.analog-stereo"

# Preference order: robot Audio mic (works) → robot Camera mic → laptop fallback.
_MIC_PREFERENCE = ("Reachy_Mini_Audio", "Reachy_Mini_Camera")

# How long to wait, at import, for the robot mic to appear in PipeWire. This is
# the heart of the "robot forgot the mic" fix: if the cable was just plugged in,
# PipeWire was restarted, or the source is still settling, the robot mic may not
# be enumerated for a second or two. Without this wait we'd silently fall back to
# the laptop mic for the WHOLE session. Polling until it shows up makes detection
# survive any audio-config change on the machine.
_MIC_DETECT_WAIT_S = 8.0


def _list_input_sources() -> list[str]:
    """Current non-monitor alsa_input source names, or [] if pactl fails."""
    try:
        out = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True, text=True, timeout=3,
        ).stdout
    except Exception:
        return []
    names = []
    for ln in out.splitlines():
        cols = ln.split("\t")
        if len(cols) >= 2 and cols[1].startswith("alsa_input") and ".monitor" not in cols[1]:
            names.append(cols[1])
    return names


def _unsuspend(source: str) -> None:
    """Ask PipeWire to wake a suspended source so the first capture isn't dropped.
    Best-effort: silently ignored if pactl/the source can't be woken."""
    try:
        subprocess.run(["pactl", "suspend-source", source, "0"],
                       capture_output=True, timeout=3)
    except Exception:
        pass


def _detect_robot_mic(default: str, wait_s: float = _MIC_DETECT_WAIT_S) -> str:
    """Return the PipeWire source name of the robot's working mic, else `default`.

    Polls the source list for up to `wait_s` seconds so a mic that PipeWire
    hasn't enumerated yet (just plugged in / PipeWire restarting / suspended)
    is still found instead of silently falling back to the laptop mic. The
    chosen source is un-suspended before returning so the first capture works.
    """
    deadline = time.time() + max(0.0, wait_s)
    while True:
        sources = _list_input_sources()
        for want in _MIC_PREFERENCE:
            for name in sources:
                if want in name:
                    _unsuspend(name)
                    return name
        if time.time() >= deadline:
            break
        # Wait a beat and re-poll — the robot mic may still be appearing.
        time.sleep(0.5)
    return default


def redetect_mic() -> str:
    """Re-run mic detection at runtime and update the module-level MIC.

    Call this if the audio config changed mid-session (e.g. the cable was
    replugged) and capture stopped working. Returns the (possibly new) MIC.
    Note: code that imported MIC by value won't see the change — read
    audio.MIC, or pass the returned value explicitly.
    """
    global MIC
    MIC = _detect_robot_mic(_LAPTOP_MIC_FALLBACK)
    return MIC


MIC = _detect_robot_mic(_LAPTOP_MIC_FALLBACK)   # robot voice mic, auto-detected (waits for it)


def startup_device_report() -> list[str]:
    """
    Return log lines describing which audio devices are active and whether the
    mic is actually producing signal. Call once at startup and log every line.
    """
    import os
    lines = []

    # ── Which devices were selected ──────────────────────────────────────────
    lines.append(f"  MIC     : {MIC}")
    lines.append(f"  SPEAKER : {SPEAKER}")

    # ── Warn if we fell back to a non-ideal device ───────────────────────────
    if MIC == _LAPTOP_MIC_FALLBACK:
        lines.append("  WARNING MIC: robot mic not found — using LAPTOP mic (room noise, wrong device!)")
    elif "Reachy_Mini_Camera" in MIC:
        lines.append("  WARNING MIC: using Camera mic — Audio mic missing (Camera mic is silent on this unit)")

    # ── USB device presence ──────────────────────────────────────────────────
    ttyACM = "/dev/ttyACM0"
    if os.path.exists(ttyACM):
        lines.append(f"  USB motors : {ttyACM} present (robot connected)")
    else:
        lines.append(f"  USB motors : {ttyACM} MISSING — robot not connected or switch wrong")

    # ── Live RMS signal test (0.5 s capture, silence = hardware issue) ───────
    try:
        proc = subprocess.Popen(
            ["pacat", "--record", "--raw",
             f"--device={MIC}",
             f"--rate={MIC_RATE}", "--channels=1", "--format=s16le"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        n = MIC_RATE // 2  # 0.5 seconds of samples
        raw = proc.stdout.read(n * 2)
        proc.terminate()
        proc.wait()
        if len(raw) >= n * 2:
            arr = np.frombuffer(raw[: n * 2], dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(arr ** 2)))
            if rms < 5:
                lines.append(
                    f"  MIC RMS : {rms:.0f} — WARNING: signal is silent! "
                    "Try replug USB cable or restart PipeWire: "
                    "`systemctl --user restart pipewire pipewire-pulse`"
                )
            elif rms < 50:
                lines.append(f"  MIC RMS : {rms:.0f} — very low (no one speaking? might be OK)")
            else:
                lines.append(f"  MIC RMS : {rms:.0f} — OK")
        else:
            lines.append(f"  MIC RMS : could not capture (only {len(raw)} bytes — pacat failed)")
    except Exception as e:
        lines.append(f"  MIC RMS : error — {e}")

    return lines


# ── Orphan-capture cleanup & mic health gate ──────────────────────────────────
# The #1 cause of "the robot doesn't listen" is a leftover `pacat --record` /
# `parecord` / `arecord` process from a crashed or kill -9'd previous run. It
# holds the robot's mic open, so the next demo's pacat gets 0 bytes of audio
# and the VAD never fires — the robot silently never hears anyone. This is the
# mic-side equivalent of the orphan reachy-mini-daemon that daemon.py already
# cleans up. Call cleanup_orphan_capture() at startup, before opening the mic.

def cleanup_orphan_capture() -> int:
    """Kill leftover mic-capture processes from crashed demo runs.
    Safe: only kills *capture* processes (`pacat --record`, `parecord`,
    `arecord`), never playback (aplay/paplay/pacat-without---record), and
    never this process itself. Returns the number of processes killed.
    """
    import os
    import signal
    me = os.getpid()
    killed = 0
    for pat in ("pacat --record", "parecord", "arecord"):
        try:
            out = subprocess.run(
                ["pgrep", "-f", pat], capture_output=True, text=True, timeout=3,
            ).stdout
        except Exception:
            continue
        for line in out.split():
            try:
                pid = int(line)
            except ValueError:
                continue
            if pid == me:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
    if killed:
        time.sleep(0.3)  # let PipeWire release the device
    return killed


def verify_mic(duration_s: float = 0.5) -> dict:
    """Open the robot mic, capture `duration_s` seconds, report signal health.
    Returns {ok, rms, bytes, reason}. `ok` is True only when the mic delivered
    full audio with RMS >= 5 (real signal — not necessarily speech, but not
    dead-silent). A short/0-byte read means the device is held by another
    process or unplugged. Does NOT clean orphans itself — call
    cleanup_orphan_capture() first if you want that.
    """
    n_samples = int(MIC_RATE * duration_s)
    n_bytes = n_samples * 2
    try:
        proc = subprocess.Popen(
            ["pacat", "--record", "--raw", f"--device={MIC}",
             f"--rate={MIC_RATE}", "--channels=1", "--format=s16le"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            raw = proc.stdout.read(n_bytes)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    except Exception as e:
        return {"ok": False, "rms": 0.0, "bytes": 0, "reason": f"capture error: {e}"}

    if len(raw) < n_bytes:
        return {"ok": False, "rms": 0.0, "bytes": len(raw),
                "reason": (f"mic delivered {len(raw)}/{n_bytes} bytes — device is "
                           "busy or unplugged. Run cleanup_orphan_capture() or "
                           "replug the USB cable, then restart.")}
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(arr ** 2)))
    if rms < 5:
        return {"ok": False, "rms": rms, "bytes": len(raw),
                "reason": (f"mic is silent (RMS {rms:.0f}) — no signal. Check the "
                           "USB cable, robot power, and the back switch (Robot mode).")}
    return {"ok": True, "rms": rms, "bytes": len(raw), "reason": "ok"}


def assert_mic_ok() -> dict:
    """Hard gate for talking demos: clean orphans, verify the mic, retry once
    on failure, then RAISE RuntimeError if the mic is truly dead (0 bytes /
    capture error). Low RMS (quiet room, no one speaking) does NOT raise —
    only a genuinely inaccessible mic does. Returns the final verify dict so
    the caller can log the RMS. Call this at startup, before listener.start().
    """
    cleanup_orphan_capture()
    info = verify_mic()
    if info["ok"]:
        return info
    # First attempt failed — try one more cleanup + verify in case a stubborn
    # orphan needed a moment to release the device.
    cleanup_orphan_capture()
    info = verify_mic()
    if info["ok"]:
        return info
    raise RuntimeError(
        f"Robot microphone is not producing audio: {info['reason']}\n"
        "The demo cannot listen without a working mic. Fix the issue above and "
        "restart. Diagnostic: ./run.sh tools/test_mic.py")


# ── Music/speaker audio detection ──────────────────────────────────────────────
# Detects whether the robot speaker is producing sound by recording a SHORT
# burst from a SECOND microphone (laptop built-in mic) and measuring RMS.
# The laptop mic is on a different ALSA device so it doesn't conflict with the
# robot mic that the VAD listener holds open during conversation demos.
#
# Usage in dance loop:
#   wait_for_music_end(MIC_FALLBACK, timeout=30)
#   → blocks until no audio detected for ~0.5 s, or timeout.
#   Returns True if music ended, False if timeout.

MIC_FALLBACK = "alsa_input.pci-0000_00_1f.3.analog-stereo"  # laptop built-in mic
_DETECT_RATE  = 16000
_DETECT_CHUNK = 1024        # 64 ms
_MUSIC_RMS_THRESH = 15      # RMS below this = silence (tune empirically)
_SILENCE_CONFIRM_MS = 600   # ms of consecutive silence → music truly ended


def wait_for_music_end(mic_device: str | None = None,
                       timeout_s: float = 30.0,
                       log_func: callable = print) -> bool:
    """
    Record from `mic_device` (defaults to laptop mic fallback) and monitor RMS
    levels. Blocks until the audio level stays below MUSIC_RMS_THRESH for
    SILENCE_CONFIRM_MS — meaning the speaker has stopped producing sound.

    Uses the laptop's built-in mic (different USB device from the robot mic),
    so it does NOT conflict with the ContinuousListener's pacat on the robot mic.

    Returns True when silence confirmed (music ended).
    Returns False on timeout.
    """
    device = mic_device or MIC_FALLBACK
    n_samples = int(_DETECT_RATE * _SILENCE_CONFIRM_MS / 1000)
    n_chunks = max(1, n_samples // _DETECT_CHUNK)
    silence_chunks = 0
    deadline = time.time() + timeout_s

    try:
        proc = subprocess.Popen(
            ["pacat", "--record", "--raw", f"--device={device}",
             f"--rate={_DETECT_RATE}", "--channels=1", "--format=s16le"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        log_func(f"  [music-detect] cannot open {device}: {e}")
        log_func("  [music-detect] falling back to process check")
        return False

    try:
        while time.time() < deadline:
            raw = proc.stdout.read(_DETECT_CHUNK * 2)
            if not raw or len(raw) < _DETECT_CHUNK * 2:
                log_func("  [music-detect] capture stream died — assuming silence")
                return True

            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(arr ** 2)))

            if rms < _MUSIC_RMS_THRESH:
                silence_chunks += 1
                if silence_chunks >= n_chunks:
                    log_func(f"  [music-detect] silence confirmed (RMS {rms:.0f})")
                    return True
            else:
                silence_chunks = 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    log_func("  [music-detect] timeout — no silence detected")
    return False


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
        subprocess.run(cmd, check=False, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(cmd, stderr=subprocess.DEVNULL)


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
    Background thinking sounds. Randomly selects from several 'phrases' so it
    never sounds mechanical. Stops within ~40ms of stop_event being set.
    """
    def _wait(secs: float) -> bool:
        end = time.time() + secs
        while time.time() < end:
            if stop_event.is_set():
                return False
            time.sleep(min(0.04, end - time.time()))
        return True

    def _run():
        import random
        rng = random.Random()

        def _rising_scan():
            chirp_nb(280, rng.randint(1000, 1300), 0.52, 0.35)
            _wait(0.57)

        def _falling_scan():
            chirp_nb(rng.randint(1100, 1400), 300, 0.44, 0.30)
            _wait(0.49)

        def _double_sweep():
            chirp_nb(420, 1050, 0.22, 0.32)
            if not _wait(0.27): return
            chirp_nb(1050, 420, 0.20, 0.25)
            _wait(0.24)

        def _data_burst():
            # 3-4 rapid blips with shifting pitch — "processing data"
            base = rng.randint(550, 950)
            freqs = [base, int(base * 1.3), base, int(base * 0.8)][:rng.randint(3, 4)]
            for f in freqs:
                if stop_event.is_set(): return
                blip(f, 0.034, 0.25, block=True)
                if not _wait(0.026): return

        def _scale_run():
            # 5-note ascending or descending flurry
            steps = [360, 480, 620, 800, 1020]
            if rng.random() > 0.5:
                steps = list(reversed(steps))
            for f in steps:
                if stop_event.is_set(): return
                blip(f, 0.036, 0.24, block=True)
                if not _wait(0.022): return

        def _wobble_pulse():
            # tremolo: amplitude-modulated tone — sounds like robot humming to itself
            freq = rng.choice([480, 580, 700, 840])
            rate = rng.choice([6, 8, 10])
            expr = f"sin(2*PI*{freq}*t)*(0.55+0.45*sin(2*PI*{rate}*t))"
            _beep(expr, 0.50, vol=0.30, block=False)
            _wait(0.55)

        def _stutter_burst():
            # rapid-fire tiny blips — like a CPU spiking
            freq = rng.randint(580, 1050)
            count = rng.randint(4, 8)
            for _ in range(count):
                if stop_event.is_set(): return
                blip(freq, 0.020, 0.20, block=True)
                if not _wait(0.015): return
                freq = int(freq * rng.uniform(0.88, 1.14))

        def _ping_echo():
            # bright ping then a softer echo a beat later
            f = rng.randint(900, 1400)
            blip(f, 0.05, 0.35, block=True)
            if not _wait(0.18): return
            blip(int(f * 0.75), 0.06, 0.18, block=True)

        def _fm_warble():
            # frequency-modulated tone — eerie wobbling sweep
            fc = rng.randint(400, 700)
            depth = rng.randint(80, 180)
            rate = rng.randint(4, 9)
            expr = f"sin(2*PI*({fc}+{depth}*sin(2*PI*{rate}*t))*t)"
            _beep(expr, 0.48, vol=0.28, block=False)
            _wait(0.52)

        phrases = [
            _rising_scan, _falling_scan, _double_sweep,
            _data_burst, _scale_run, _wobble_pulse,
            _stutter_burst, _ping_echo, _fm_warble,
        ]
        weights = [4, 3, 3, 4, 3, 2, 3, 2, 2]

        while not stop_event.is_set():
            rng.choices(phrases, weights=weights, k=1)[0]()
            if stop_event.is_set():
                break
            if not _wait(rng.uniform(0.55, 1.20)):
                break
        # Kill any in-flight ffmpeg tones — non-blocking chirps/beeps may still
        # be playing their full duration (~100-500ms) after the loop exits.
        # Killing them frees the speaker immediately so TTS starts cleanly.
        subprocess.run(["pkill", "-9", "-f", "aevalsrc"], check=False)

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
    chirp(820, 430, 0.32, vol=0.50, block=False)

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
                # Mic stream died (0 bytes) — device lost or grabbed by another
                # process. Surface it instead of silently returning None, which
                # would make the demo loop forever showing "Listening..." with
                # no response. The caller's finally still runs goto_sleep().
                raise RuntimeError(
                    f"Microphone stream closed unexpectedly "
                    f"(got {len(raw) if raw else 0} bytes). The mic device may "
                    "have been unplugged or grabbed by another process. "
                    "Run cleanup_orphan_capture() and replug USB, then restart."
                )

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
