"""
demo_dance.py — Network School Full Show (Macarena Edition)
===========================================================
Record cue → boot → IMMEDIATE wake-up → greeting speech →
beat-synced Macarena choreography + music → climax → bow → sleep.

Beat-sync: pre-analyzed at 103.4 BPM (0.5805 s/beat).
Body sweeps up to ±1.4 rad (80°). Jump move: slow push-down → fast snap-up.

Run:  ./run.sh demos/demo_dance.py
"""
import math
import subprocess
import time
import wave
from pathlib import Path

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose

from reachy_demo.daemon import start_daemon, stop_daemon
from reachy_demo.tts_edge import synth_to_file

ROOT    = Path(__file__).parent.parent
SPEAKER = "plughw:CARD=Audio,DEV=0"

# ── Music ─────────────────────────────────────────────────────────────────
# Swap this one line to use any MP3 from the music/ folder.
MUSIC = str(ROOT / "music" / "macarena.mp3")
# MUSIC = str(ROOT / "music" / "blipotron.mp3")
# ─────────────────────────────────────────────────────────────────────────

GREETING = (
    "Welcome to Network School! "
    "What would you like to talk about? "
    "Robotics, Artificial Intelligence, Crypto, or Network States?"
)
TEASER = "And now... watch this!"

# Beat timing pre-analyzed from macarena.mp3 with librosa (103.4 BPM)
BEAT = 0.5805   # seconds per beat

# ---------------------------------------------------------------------------
# Sound effects
# ---------------------------------------------------------------------------

def _play(expr, duration, vol=0.7):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"aevalsrc={expr}*{vol}:c=mono:s=22050",
         "-t", str(duration), "-f", "alsa", SPEAKER],
        check=False,
    )

def chirp(f0, f1, dur, vol=0.65):
    _play(f"sin(2*PI*({f0}*t+({f1}-{f0})*t*t/(2*{dur})))", dur, vol)

def blip(freq, dur=0.08, vol=0.5):
    _play(f"sin(2*PI*{freq}*t)", dur, vol)

def record_cue():
    for _ in range(3):
        chirp(600, 1000, 0.12, vol=0.8)
        time.sleep(0.6)

def boot_sequence():
    chirp(300,  900, 0.18); time.sleep(0.05)
    chirp(600, 1400, 0.14); time.sleep(0.04)
    chirp(900,  400, 0.20); time.sleep(0.04)
    blip(1200, 0.06);       time.sleep(0.03)
    blip(1600, 0.07)

def excited_chirp():
    chirp(500, 1800, 0.14, vol=0.75)
    time.sleep(0.04)
    chirp(800, 2200, 0.12, vol=0.85)

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def synth(text: str):
    """Synthesise text via edge-tts. Returns (duration_s, wav_path)."""
    path = synth_to_file(text)
    with wave.open(path) as wf:
        duration = wf.getnframes() / wf.getframerate()
    return duration, path

# ---------------------------------------------------------------------------
# Greeting animation (layered sine waves — looks organic)
# ---------------------------------------------------------------------------

STYLES = {
    "welcome": dict(
        ph=(0.14, 0.50), ph2=(0.05, 1.31),
        ya=(0.22, 0.31), ya2=(0.07, 0.73),
        ro=(0.08, 0.19), ro2=(0.03, 0.53),
        an=(0.55, 0.61), an2=(0.15, 1.17),
        by=(0.70, 0.17), by2=(0.20, 0.41),
    ),
    "talk": dict(
        ph=(0.10, 0.50), ph2=(0.04, 1.27),
        ya=(0.18, 0.27), ya2=(0.05, 0.71),
        ro=(0.05, 0.15), ro2=(0.02, 0.43),
        an=(0.42, 0.55), an2=(0.12, 1.09),
        by=(0.60, 0.13), by2=(0.18, 0.37),
    ),
    "curious": dict(
        ph=(0.08, 0.35), ph2=(0.03, 0.89),
        ya=(0.20, 0.21), ya2=(0.06, 0.59),
        ro=(0.12, 0.29), ro2=(0.04, 0.67),
        an=(0.32, 0.43), an2=(0.10, 0.97),
        by=(0.55, 0.11), by2=(0.16, 0.31),
    ),
}

def _w(c, key, t):
    a1, f1 = c[key]; a2, f2 = c[key + "2"]
    return a1 * math.sin(2 * math.pi * f1 * t) + a2 * math.sin(2 * math.pi * f2 * t)

def speak_and_animate(mini, audio_path, audio_duration):
    proc = subprocess.Popen(
        ["aplay", "-D", SPEAKER, "-q", audio_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    cuts = [audio_duration * 0.28, audio_duration * 0.52]
    t0 = time.time()
    while proc.poll() is None:
        t = time.time() - t0
        style = "talk" if t >= cuts[1] else ("curious" if t >= cuts[0] else "welcome")
        c = STYLES[style]
        mini.set_target(
            head=create_head_pose(
                pitch=_w(c, "ph", t), yaw=_w(c, "ya", t),
                roll=_w(c, "ro", t), degrees=False,
            ),
            antennas=[_w(c, "an", t), -_w(c, "an", t)],
            body_yaw=_w(c, "by", t),
        )
        time.sleep(0.05)
    proc.wait()

# ---------------------------------------------------------------------------
# Music
# ---------------------------------------------------------------------------

def play_music(path: str) -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-stream_loop", "-1", "-i", path,
         "-af", "volume=0.85", "-f", "alsa", SPEAKER],
    )

# ---------------------------------------------------------------------------
# Beat-synced dance moves
# ---------------------------------------------------------------------------

# 8-pose Macarena cycle — each pose held for one beat (0.58 s).
# Mimics the iconic arm sequence: right out → left out → right shoulder →
# left shoulder → cross → cross → hands-to-head → hip shake.
# Scale factor increases each cycle so moves get bigger.
MACARENA_POSES = [
    # pitch   yaw     roll    body_yaw  [ant_L, ant_R]
    ( 0.08, -0.38,  0.08,   0.50, [ 0.15, -0.55]),  # 0 right arm out
    ( 0.13, -0.48,  0.12,   0.75, [ 0.08, -0.72]),  # 1 right arm up
    ( 0.08,  0.38, -0.08,  -0.50, [ 0.55, -0.15]),  # 2 left arm out
    ( 0.13,  0.48, -0.12,  -0.75, [ 0.72, -0.08]),  # 3 left arm up
    ( 0.04, -0.18,  0.28,   0.95, [ 0.55, -0.55]),  # 4 right shoulder cross
    ( 0.04,  0.18, -0.28,  -0.95, [-0.55,  0.55]),  # 5 left shoulder cross
    (-0.20,  0.00,  0.12,   1.25, [ 0.75,  0.75]),  # 6 hands to head
    (-0.12,  0.04, -0.12,  -1.40, [ 0.65,  0.65]),  # 7 hip shake left
]

def macarena_beat(mini, pose, scale=1.0):
    p, y, r, by, ants = pose
    clamp = lambda v, lim: max(-lim, min(lim, v))
    mini.goto_target(
        head=create_head_pose(
            pitch=clamp(p * scale, 0.38),
            yaw=clamp(y * scale, 1.57),
            roll=clamp(r * scale, 0.38),
            degrees=False,
        ),
        antennas=[ants[0], ants[1]],
        body_yaw=clamp(by * scale, 1.40),
        duration=BEAT - 0.06,
    )
    time.sleep(0.06)


def jump(mini):
    """Slow push-down → instant snap-up (slingshot effect)."""
    print("    ↓ JUMP ↓")
    mini.goto_target(
        head=create_head_pose(pitch=-0.36, roll=0.08, degrees=False),
        body_yaw=0.0, duration=0.55,
    )
    time.sleep(0.02)
    mini.goto_target(
        head=create_head_pose(pitch=0.38, roll=-0.05, degrees=False),
        antennas=[0.8, 0.8], body_yaw=0.0, duration=0.07,
    )
    time.sleep(0.10)


def macarena_section(mini, cycles=3):
    """Run the Macarena cycle, escalating each pass, with a jump at the end of each cycle."""
    for c in range(cycles):
        scale = 1.0 + c * 0.18          # 1.0 → 1.18 → 1.36
        print(f"  -- Macarena cycle {c+1} (scale {scale:.2f}) --")
        for i, pose in enumerate(MACARENA_POSES):
            macarena_beat(mini, pose, scale)
        if c > 0:
            jump(mini)


def spin(mini, angle, duration=0.45):
    """Hard body-turn to angle (radians) — visible dramatic transition."""
    mini.goto_target(
        head=create_head_pose(),
        antennas=[0.0, 0.0],
        body_yaw=angle, duration=duration,
    )
    time.sleep(duration + 0.05)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Network School — Macarena Show")

    print("  Generating speech...")
    greet_dur, WAV_GREET = synth(GREETING)
    tease_dur, WAV_TEASE = synth(TEASER)
    print(f"  Greeting: {greet_dur:.1f}s   Teaser: {tease_dur:.1f}s")

    print("\n  >>> RECORD CUE — hit record! <<<")
    record_cue()

    # Boot immediately after cue — no gap
    print("  Boot sequence...")
    boot_sequence()

    print("  Starting daemon...")
    daemon_proc = start_daemon()

    try:
        em = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        da = RecordedMoves("pollen-robotics/reachy-mini-dances-library")

        with ReachyMini(connection_mode="localhost_only",
                        media_backend="no_media",
                        spawn_daemon=False) as mini:

            mini.wake_up()

            # ── Act 1: Greeting ──────────────────────────────────────────
            print("\n  ── Act 1: Greeting ──")
            speak_and_animate(mini, WAV_GREET, greet_dur)
            speak_and_animate(mini, WAV_TEASE, tease_dur)
            excited_chirp()

            # ── Act 2: Macarena ──────────────────────────────────────────
            print("\n  ── Act 2: Macarena ──")
            beat = play_music(MUSIC)
            try:
                # Dramatic entry spin
                spin(mini,  1.4, duration=0.35)
                spin(mini, -1.4, duration=0.35)
                spin(mini,  0.0, duration=0.30)

                # 3 escalating Macarena cycles
                macarena_section(mini, cycles=3)

                # Climax sweep + big preset
                print("  *** CLIMAX ***")
                spin(mini,  1.4, duration=0.30)
                mini.play_move(em.get("dance3"), play_frequency=80.0, sound=False)
                spin(mini, -1.4, duration=0.30)
                mini.play_move(em.get("success2"), play_frequency=80.0, sound=False)

            finally:
                beat.terminate()
                beat.wait()

            # ── Bow out ──────────────────────────────────────────────────
            print("\n  ── Bow out ──")
            spin(mini, 0.0, duration=0.4)
            mini.play_move(em.get("loving1"), play_frequency=80.0, sound=False)

            mini.goto_target(
                head=create_head_pose(), antennas=[0.0, 0.0],
                duration=1.0, body_yaw=0.0,
            )
            time.sleep(1.1)
            mini.goto_sleep()
            print("  Show complete!")

    finally:
        Path(WAV_GREET).unlink(missing_ok=True)
        Path(WAV_TEASE).unlink(missing_ok=True)
        stop_daemon(daemon_proc)


if __name__ == "__main__":
    main()
