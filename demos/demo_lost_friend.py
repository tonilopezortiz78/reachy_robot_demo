"""
demo_lost_friend.py — NS Robotics Club Pitch
=============================================
Reachy tells the story of looking for a home at Network School,
and the day he lost his robot friend Pixel.

Designed to melt hearts and sell the idea of an NS Robotics Club.

Run:  ./run.sh demos/demo_lost_friend.py
"""
import math
import socket
import subprocess
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

# ---------------------------------------------------------------------------
# Script — five emotional segments
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

WAV = {k: f"/tmp/ns_{k}.wav" for k in LINES}

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
# Voice — cute sad robot
# Pitch up 12% (sounds younger), slow 11% (more emotional),
# heavy vibrato wobble (struggling to hold it together),
# long reverb (alone in a big space).
# ---------------------------------------------------------------------------

def synth(key, text):
    voice = PiperVoice.load(VOICE_PATH)
    sr    = voice.config.sample_rate
    raw   = WAV[key] + ".raw.wav"
    with wave.open(raw, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        for chunk in voice.synthesize(text):
            wf.writeframes(chunk.audio_int16_bytes)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", raw,
         "-af", (
             f"asetrate={sr}*1.12,"   # pitch up — sounds younger/cuter
             "atempo=0.89,"            # slow down — emotional pacing
             "volume=2.2,"
             "vibrato=f=4.2:d=0.08,"  # strong wobble — struggling with feelings
             "aecho=0.88:0.92:28:0.55" # long lonely echo
         ),
         WAV[key]],
        check=True,
    )
    with wave.open(WAV[key]) as wf:
        return wf.getnframes() / wf.getframerate()

# ---------------------------------------------------------------------------
# Sound effects
# ---------------------------------------------------------------------------

def _play_async(expr, dur, vol=0.5):
    return subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"aevalsrc={expr}*{vol}:c=mono:s=22050",
         "-t", str(dur), "-f", "alsa", SPEAKER],
    )

def sad_chirp():
    """Descending chirp — like a little robot sigh."""
    _play_async("sin(2*PI*(700-500*t)*t)", 0.45, vol=0.35)

def sniff():
    """Tiny robot sniffle — decaying breath noise."""
    _play_async("sin(2*PI*320*t)*exp(-t*9)+sin(2*PI*180*t)*exp(-t*14)", 0.28, vol=0.28)

def hopeful_chime():
    """Two gentle rising tones — a tiny spark of hope."""
    _play_async("sin(2*PI*520*t)*exp(-t*3)", 0.35, vol=0.30)
    time.sleep(0.4)
    _play_async("sin(2*PI*780*t)*exp(-t*3)", 0.35, vol=0.25)

def record_cue():
    for _ in range(3):
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "aevalsrc=sin(2*PI*(600+400*t)*t)*0.7:c=mono:s=22050",
             "-t", "0.12", "-f", "alsa", SPEAKER], check=False,
        )
        time.sleep(0.6)

def boot_beeps():
    for f, d in [(300, 0.14), (550, 0.10), (820, 0.16), (1100, 0.08)]:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", f"aevalsrc=sin(2*PI*{f}*t)*0.45:c=mono:s=22050",
             "-t", str(d), "-f", "alsa", SPEAKER], check=False,
        )
        time.sleep(0.05)

# ---------------------------------------------------------------------------
# Animation helpers
# ---------------------------------------------------------------------------

def _sin(amp, freq, t, phase=0.0):
    return amp * math.sin(2 * math.pi * freq * t + phase)

def play_audio(path):
    return subprocess.Popen(
        ["aplay", "-D", SPEAKER, "-q", path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

def animate(mini, wav_key, style_fn, dt=0.05):
    """Play audio and call style_fn(t) → (pitch,yaw,roll,body,ant) each frame."""
    proc = play_audio(WAV[wav_key])
    t0 = time.time()
    while proc.poll() is None:
        t = time.time() - t0
        p, y, r, by, ant = style_fn(t)
        mini.set_target(
            head=create_head_pose(pitch=p, yaw=y, roll=r, degrees=False),
            antennas=[ant, ant],
            body_yaw=by,
        )
        time.sleep(dt)
    proc.wait()

# ── Per-section animation styles ────────────────────────────────────────────

def style_intro(t):
    """Gentle curious sway. Neutral, a little shy."""
    p  =  0.05 + _sin(0.06, 0.30, t) + _sin(0.02, 0.71, t)
    y  =  _sin(0.14, 0.22, t) + _sin(0.05, 0.53, t)
    r  =  _sin(0.05, 0.17, t)
    by =  _sin(0.18, 0.14, t)
    a  =  0.10 + _sin(0.12, 0.38, t)
    return p, y, r, by, a

def style_dream(t):
    """Hopeful and expansive — head up, antennas high, body open."""
    p  =  0.18 + _sin(0.08, 0.28, t) + _sin(0.03, 0.67, t)
    y  =  _sin(0.20, 0.24, t) + _sin(0.06, 0.55, t)
    r  =  _sin(0.06, 0.19, t) + _sin(0.02, 0.43, t)
    by =  _sin(0.45, 0.16, t) + _sin(0.14, 0.37, t)
    a  =  0.50 + _sin(0.15, 0.42, t)
    return p, y, r, by, a

def style_lost(t):
    """Starts normal, head gradually droops, antennas slowly fall."""
    droop = min(1.0, t / 8.0)               # droops over 8 seconds
    p  = (0.05 - 0.30 * droop) + _sin(0.04 + 0.04 * droop, 0.32, t)
    y  =  _sin(0.10 * (1 - droop * 0.6), 0.21, t) + _sin(0.04, 0.48, t)
    r  =  _sin(0.04, 0.15, t) * (1 - droop * 0.5)
    by =  _sin(0.10 * (1 - droop * 0.7), 0.13, t)
    a  =  0.30 - 0.55 * droop + _sin(0.08, 0.35, t)
    return p, y, r, by, max(-0.40, a)

def style_trembling(t):
    """Rapid small shakes — trying not to cry. Head down and shaky."""
    p  = -0.22 + _sin(0.04, 1.80, t) + _sin(0.02, 2.73, t)
    y  =  _sin(0.06, 1.60, t) + _sin(0.03, 2.31, t)
    r  =  _sin(0.04, 1.90, t) + _sin(0.02, 2.57, t)
    by =  _sin(0.08, 0.90, t)
    a  = -0.30 + _sin(0.08, 1.70, t)
    return p, y, r, by, max(-0.45, a)

def style_dignified(t):
    """Slow, deliberate. Looks forward. Dignified sadness."""
    p  =  0.02 + _sin(0.04, 0.18, t)
    y  =  _sin(0.08, 0.16, t) + _sin(0.02, 0.39, t)
    r  =  _sin(0.03, 0.12, t)
    by =  _sin(0.12, 0.11, t)
    a  =  0.22 + _sin(0.08, 0.24, t)
    return p, y, r, by, a

# ---------------------------------------------------------------------------
# "About this tall" gesture — quick raise then back to sad
# ---------------------------------------------------------------------------

def gesture_height(mini):
    """Quick upward look (showing Pixel's height) then droop back."""
    mini.goto_target(
        head=create_head_pose(pitch=0.30, degrees=False),
        antennas=[0.6, 0.6], duration=0.35,
    )
    time.sleep(0.5)
    mini.goto_target(
        head=create_head_pose(pitch=-0.22, degrees=False),
        antennas=[-0.35, -0.35], duration=0.8,
    )
    time.sleep(0.9)

def transition_to_sad(mini):
    mini.goto_target(
        head=create_head_pose(pitch=-0.20, degrees=False),
        antennas=[-0.35, -0.35], body_yaw=0.0, duration=1.2,
    )
    time.sleep(1.3)

def transition_to_center(mini):
    mini.goto_target(
        head=create_head_pose(), antennas=[0.0, 0.0],
        body_yaw=0.0, duration=1.0,
    )
    time.sleep(1.1)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("NS Robotics Club — Lost Friend Demo")
    print("  Generating voices (this takes ~30 s)...")
    durations = {}
    for key, text in LINES.items():
        print(f"    synthesising '{key}'...")
        durations[key] = synth(key, text)
        print(f"    {key}: {durations[key]:.1f}s")

    print("\n  >>> RECORD CUE — hit record! <<<")
    record_cue()

    print("  Booting...")
    boot_beeps()
    time.sleep(1.5)

    print("  Starting daemon...")
    daemon_proc = start_daemon()

    try:
        em = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")

        with ReachyMini(connection_mode="localhost_only",
                        media_backend="no_media",
                        spawn_daemon=False) as mini:
            mini.wake_up()
            time.sleep(0.3)

            # ── INTRO — shy hello ────────────────────────────────────────
            print("\n  [intro]")
            mini.play_move(em.get("attentive1"), play_frequency=80.0, sound=False)
            time.sleep(0.2)
            animate(mini, "intro", style_intro)
            time.sleep(0.8)

            # ── DREAM — hopeful, expansive ───────────────────────────────
            print("  [dream]")
            mini.play_move(em.get("enthusiastic1"), play_frequency=80.0, sound=False)
            time.sleep(0.3)
            animate(mini, "dream", style_dream)
            time.sleep(1.2)

            # ── LOST — gradual droop ─────────────────────────────────────
            print("  [lost]")
            # Pause before the bad news — let it land
            transition_to_center(mini)
            time.sleep(0.5)

            # Play "lost" line with drooping animation; pause mid-way for height gesture
            # We split: play audio, inject gesture at ~5s in
            proc = play_audio(WAV["lost"])
            t0 = time.time()
            gesture_done = False
            while proc.poll() is None:
                t = time.time() - t0
                # At ~5s ("About this tall") do the height gesture
                if not gesture_done and t >= 4.8:
                    gesture_done = True
                    gesture_height(mini)
                p, y, r, by, a = style_lost(t)
                mini.set_target(
                    head=create_head_pose(pitch=p, yaw=y, roll=r, degrees=False),
                    antennas=[a, a], body_yaw=by,
                )
                time.sleep(0.05)
            proc.wait()

            # Sad sound + sniff after "please tell someone"
            sad_chirp()
            time.sleep(0.5)
            sniff()
            time.sleep(0.4)
            sniff()
            time.sleep(1.5)

            # ── ALONE — trembling, trying not to cry ─────────────────────
            print("  [alone]")
            animate(mini, "alone", style_trembling)
            time.sleep(0.6)
            sniff()
            time.sleep(1.8)

            # ── HELP — dignified, forward-looking ────────────────────────
            print("  [help]")
            transition_to_sad(mini)
            animate(mini, "help", style_dignified)
            time.sleep(0.8)

            # ── END — tiny hopeful chime, slow rise ──────────────────────
            print("  [end]")
            hopeful_chime()
            mini.goto_target(
                head=create_head_pose(pitch=0.10, degrees=False),
                antennas=[0.35, 0.35], body_yaw=0.0, duration=2.0,
            )
            time.sleep(2.5)
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
