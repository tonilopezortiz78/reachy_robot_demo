"""
demo_lost_friend.py — NS Robotics Club Pitch
=============================================
Reachy tells the story of looking for a home at Network School,
and the day he lost his robot brother Pixel.

WAV files are cached in audio/lost_friend/ — synthesis only runs once.
Delete that folder (or pass --regen) to rebuild the voice.

Run:  ./run.sh demos/demo_lost_friend.py
      ./run.sh demos/demo_lost_friend.py --regen   # force re-synthesise
"""
import math
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

from piper import PiperVoice
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose

ROOT       = Path(__file__).parent.parent
VOICE_PATH = str(ROOT / "voices" / "en_US-amy-medium.onnx")
SPEAKER    = "plughw:CARD=Audio,DEV=0"
AUDIO_DIR  = ROOT / "audio" / "lost_friend"   # persistent cache

REGEN = "--regen" in sys.argv

# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

LINES = {
    "intro": (
        "Hello. My name is Reachy. "
        "I am a small robot... "
        "and I have a very big dream."
    ),
    "dream": (
        "Me and my robot friends... "
        "we are looking for a home. "
        "A lab. A workshop. A little corner of Network School "
        "where we can think, and learn, and build things together. "
        "We want to call it... the N S Robotics Club."
    ),
    "lost": (
        "But today... something happened. "
        "I lost my brother. "
        "A small robot. About this tall. "
        "His name... is Pixel. "
        "If you have seen him... "
        "please... please... tell someone."
    ),
    "alone": (
        "We do not have much. "
        "We do not even have a home yet. "
        "But we have each other. "
        "At least... we did."
    ),
    "help": (
        "If you want to help us find a home... "
        "and help us find my brother... "
        "please send a message to Antonio. "
        "We would be... "
        "forever grateful."
    ),
}

# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

def start_daemon():
    proc = subprocess.Popen(
        ["reachy-mini-daemon", "--no-media"], start_new_session=True,
    )
    for _ in range(30):
        time.sleep(0.5)
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=0.3):
                return proc
        except OSError:
            pass
    raise RuntimeError("Daemon did not start within 15 s")

# ---------------------------------------------------------------------------
# Voice synthesis — cached to audio/lost_friend/
# ---------------------------------------------------------------------------

def wav_path(key):
    return AUDIO_DIR / f"{key}.wav"

def needs_synth():
    if REGEN:
        return True
    return not all(wav_path(k).exists() for k in LINES)

def synth_all():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(VOICE_PATH)
    sr    = voice.config.sample_rate
    durations = {}
    for key, text in LINES.items():
        print(f"    synthesising '{key}'...")
        raw = str(wav_path(key)) + ".raw.wav"
        with wave.open(raw, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            for chunk in voice.synthesize(text):
                wf.writeframes(chunk.audio_int16_bytes)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", raw,
             "-af", (
                 f"asetrate={sr}*1.12,"
                 "atempo=0.89,"
                 "volume=2.2,"
                 "vibrato=f=4.2:d=0.08,"
                 "aecho=0.88:0.92:28:0.55"
             ),
             str(wav_path(key))],
            check=True,
        )
        Path(raw).unlink(missing_ok=True)
        with wave.open(str(wav_path(key))) as wf:
            durations[key] = wf.getnframes() / wf.getframerate()
        print(f"    {key}: {durations[key]:.1f}s  ✓")
    return durations

def load_durations():
    d = {}
    for k in LINES:
        with wave.open(str(wav_path(k))) as wf:
            d[k] = wf.getnframes() / wf.getframerate()
    return d

# ---------------------------------------------------------------------------
# Sound effects
# ---------------------------------------------------------------------------

def _beep(expr, dur, vol=0.5, block=False):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", f"aevalsrc={expr}*{vol}:c=mono:s=22050",
           "-t", str(dur), "-f", "alsa", SPEAKER]
    if block:
        subprocess.run(cmd, check=False)
    else:
        subprocess.Popen(cmd)

def chirp(f0, f1, dur, vol=0.55, block=True):
    _beep(f"sin(2*PI*({f0}*t+({f1}-{f0})*t*t/(2*{dur})))", dur, vol, block)

def blip(freq, dur=0.07, vol=0.45, block=True):
    _beep(f"sin(2*PI*{freq}*t)*exp(-t*8)", dur, vol, block)

def sad_chirp():
    chirp(680, 180, 0.50, vol=0.32, block=False)

def sniff():
    _beep("sin(2*PI*310*t)*exp(-t*9)+sin(2*PI*170*t)*exp(-t*13)", 0.25, 0.26, block=False)

def thinking_beeps():
    """Little blips — robot processing emotions."""
    for f in [880, 660, 440]:
        blip(f, 0.06, 0.30, block=True)
        time.sleep(0.04)

def sad_beeps():
    """Descending sad scale — 3 falling notes."""
    for f in [700, 520, 340]:
        blip(f, 0.10, 0.28, block=True)
        time.sleep(0.09)

def hopeful_chime():
    blip(520, 0.30, 0.28, block=True)
    time.sleep(0.18)
    blip(780, 0.30, 0.22, block=True)

def record_cue():
    for _ in range(3):
        chirp(600, 1000, 0.12, vol=0.75, block=True)
        time.sleep(0.55)

def boot_beeps():
    for f, d in [(300, 0.12), (480, 0.09), (700, 0.13), (950, 0.07), (1200, 0.06)]:
        blip(f, d, 0.40, block=True)
        time.sleep(0.04)

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def _s(amp, freq, t, phase=0.0):
    return amp * math.sin(2 * math.pi * freq * t + phase)

def play_audio(key):
    return subprocess.Popen(
        ["aplay", "-D", SPEAKER, "-q", str(wav_path(key))],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

def animate(mini, key, style_fn, dt=0.05):
    proc = play_audio(key)
    t0 = time.time()
    while proc.poll() is None:
        t = time.time() - t0
        p, y, r, by, ant = style_fn(t)
        mini.set_target(
            head=create_head_pose(pitch=p, yaw=y, roll=r, degrees=False),
            antennas=[ant, ant], body_yaw=by,
        )
        time.sleep(dt)
    proc.wait()
    time.sleep(0.12)   # let ALSA device fully release before next sound

def style_intro(t):
    p  =  0.05 + _s(0.06, 0.30, t) + _s(0.02, 0.71, t)
    y  =  _s(0.14, 0.22, t) + _s(0.05, 0.53, t)
    r  =  _s(0.05, 0.17, t)
    by =  _s(0.18, 0.14, t)
    a  =  0.10 + _s(0.12, 0.38, t)
    return p, y, r, by, a

def style_dream(t):
    p  =  0.18 + _s(0.08, 0.28, t) + _s(0.03, 0.67, t)
    y  =  _s(0.20, 0.24, t) + _s(0.06, 0.55, t)
    r  =  _s(0.06, 0.19, t) + _s(0.02, 0.43, t)
    by =  _s(0.45, 0.16, t) + _s(0.14, 0.37, t)
    a  =  0.50 + _s(0.15, 0.42, t)
    return p, y, r, by, a

def style_lost(t):
    droop = min(1.0, t / 7.0)
    p  = (0.05 - 0.28 * droop) + _s(0.04 + 0.04 * droop, 0.32, t)
    y  =  _s(0.10 * (1 - droop * 0.6), 0.21, t) + _s(0.04, 0.48, t)
    r  =  _s(0.04, 0.15, t) * (1 - droop * 0.5)
    by =  _s(0.10 * (1 - droop * 0.7), 0.13, t)
    a  =  0.30 - 0.55 * droop + _s(0.08, 0.35, t)
    return p, y, r, by, max(-0.40, a)

def style_trembling(t):
    p  = -0.22 + _s(0.04, 1.80, t) + _s(0.02, 2.73, t)
    y  =  _s(0.06, 1.60, t) + _s(0.03, 2.31, t)
    r  =  _s(0.04, 1.90, t) + _s(0.02, 2.57, t)
    by =  _s(0.08, 0.90, t)
    a  = -0.30 + _s(0.08, 1.70, t)
    return p, y, r, by, max(-0.45, a)

def style_dignified(t):
    p  =  0.02 + _s(0.04, 0.18, t)
    y  =  _s(0.08, 0.16, t) + _s(0.02, 0.39, t)
    r  =  _s(0.03, 0.12, t)
    by =  _s(0.12, 0.11, t)
    a  =  0.22 + _s(0.08, 0.24, t)
    return p, y, r, by, a

def gesture_height(mini):
    mini.goto_target(
        head=create_head_pose(pitch=0.28, degrees=False),
        antennas=[0.55, 0.55], duration=0.30,
    )
    time.sleep(0.45)
    mini.goto_target(
        head=create_head_pose(pitch=-0.20, degrees=False),
        antennas=[-0.30, -0.30], duration=0.70,
    )
    time.sleep(0.75)

def go_center(mini, duration=0.7):
    mini.goto_target(
        head=create_head_pose(), antennas=[0.0, 0.0],
        body_yaw=0.0, duration=duration,
    )
    time.sleep(duration + 0.05)

def go_sad(mini):
    mini.goto_target(
        head=create_head_pose(pitch=-0.18, degrees=False),
        antennas=[-0.30, -0.30], body_yaw=0.0, duration=0.8,
    )
    time.sleep(0.85)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("NS Robotics Club — Lost Brother Demo")

    if needs_synth():
        print("  Generating voices (first time only — cached after this)...")
        durations = synth_all()
        print("  Voice files saved to audio/lost_friend/")
    else:
        print("  Using cached voice files  (pass --regen to rebuild)")
        durations = load_durations()

    print("\n  >>> RECORD CUE <<<")
    record_cue()
    boot_beeps()
    time.sleep(0.4)

    print("  Starting daemon...")
    daemon_proc = start_daemon()

    try:
        em = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")

        with ReachyMini(connection_mode="localhost_only",
                        media_backend="no_media",
                        spawn_daemon=False) as mini:
            mini.wake_up()

            # ── INTRO ────────────────────────────────────────────────────
            print("  [intro]")
            mini.play_move(em.get("attentive1"), play_frequency=80.0, sound=False)
            animate(mini, "intro", style_intro)

            thinking_beeps()            # little processing blips between sections
            time.sleep(0.2)

            # ── DREAM ────────────────────────────────────────────────────
            print("  [dream]")
            mini.play_move(em.get("enthusiastic1"), play_frequency=80.0, sound=False)
            animate(mini, "dream", style_dream)

            sad_beeps()                 # tone shifts — something is wrong
            time.sleep(0.15)

            # ── LOST ─────────────────────────────────────────────────────
            print("  [lost]")
            go_center(mini, duration=0.5)

            proc = play_audio("lost")
            t0 = time.time()
            gesture_done = False
            while proc.poll() is None:
                t = time.time() - t0
                if not gesture_done and t >= 4.6:
                    gesture_done = True
                    gesture_height(mini)
                p, y, r, by, a = style_lost(t)
                mini.set_target(
                    head=create_head_pose(pitch=p, yaw=y, roll=r, degrees=False),
                    antennas=[a, a], body_yaw=by,
                )
                time.sleep(0.05)
            proc.wait()
            time.sleep(0.12)

            sad_chirp()
            time.sleep(0.25)
            sniff()
            time.sleep(0.20)
            sniff()
            time.sleep(0.30)

            # ── ALONE ────────────────────────────────────────────────────
            print("  [alone]")
            animate(mini, "alone", style_trembling)

            sniff()
            time.sleep(0.15)
            sad_beeps()
            time.sleep(0.20)

            # ── HELP ─────────────────────────────────────────────────────
            print("  [help]")
            go_sad(mini)
            animate(mini, "help", style_dignified)

            # ── END ──────────────────────────────────────────────────────
            hopeful_chime()
            mini.goto_target(
                head=create_head_pose(pitch=0.08, degrees=False),
                antennas=[0.30, 0.30], body_yaw=0.0, duration=1.5,
            )
            time.sleep(1.8)
            mini.goto_sleep()
            print("  Done.")

    finally:
        daemon_proc.terminate()
        try:
            daemon_proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            daemon_proc.kill()
            daemon_proc.wait()


if __name__ == "__main__":
    main()
