"""
Network School Greeter
Speaks a welcome message with natural layered motion — overlapping sine waves
at incommensurable frequencies on pitch/yaw/roll/antennas/body so no axis
ever looks mechanical.

Flow:
  boot-beeps → 3 s → wake up → welcoming preset → speak + animate → attentive → sleep

Voice processing:
  Piper TTS output is piped through ffmpeg to add a subtle robotic effect
  (gentle vibrato + quick metallic echo).  Set ROBOT_VOICE_FX=False to hear
  the raw Piper voice.
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

SPEAKER = "plughw:CARD=Audio,DEV=0"

GREETING = (
    "Welcome... to Network School! "
    "What would you like to talk about? "
    "Robotics, Artificial Intelligence, Crypto, or Network States? "
    "I am all ears!"
)

# ---------------------------------------------------------------------------
# Sound effects — all generated synthetically with ffmpeg, no downloads needed
# ---------------------------------------------------------------------------

def _play(expr: str, duration: float, vol: float = 0.7):
    """Evaluate an ffmpeg aevalsrc expression and play it on the robot speaker."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi",
         "-i", f"aevalsrc={expr}*{vol}:c=mono:s=22050",
         "-t", str(duration),
         "-f", "alsa", SPEAKER],
        check=False,
    )


def chirp(f0: float, f1: float, dur: float, vol: float = 0.65):
    """Linear frequency sweep from f0 to f1 Hz — R2-D2 style chirp."""
    # Phase of a linear chirp: φ(t) = 2π(f0·t + (f1-f0)·t²/(2·dur))
    expr = f"sin(2*PI*({f0}*t+({f1}-{f0})*t*t/(2*{dur})))"
    _play(expr, dur, vol)


def blip(freq: float, dur: float = 0.08, vol: float = 0.5):
    """Short pure-tone blip."""
    _play(f"sin(2*PI*{freq}*t)", dur, vol)


def record_cue():
    """
    Three slow beeps — the classic 'start recording now' signal.
    Each beep is a short rising chirp so it's unmistakeable on the recording.
    After the third beep the caller should wait ~1 s before starting the demo.
    """
    for i in range(3):
        chirp(600, 1000, 0.12, vol=0.8)
        time.sleep(0.6)


def boot_sequence():
    """Ascending R2-D2-style startup chirps."""
    chirp(300, 900, 0.18)
    time.sleep(0.06)
    chirp(600, 1400, 0.14)
    time.sleep(0.04)
    chirp(900, 400, 0.20)     # descending swoosh
    time.sleep(0.05)
    # Double affirmative blip
    blip(1200, 0.06)
    time.sleep(0.04)
    blip(1600, 0.06)


def ready_blip():
    """Two-tone 'ready' chime."""
    blip(880, 0.10)
    time.sleep(0.07)
    blip(1320, 0.12)


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
# Natural layered animation
# ---------------------------------------------------------------------------
# Each axis uses two overlapping sine waves at irrational-ratio frequencies
# so the pattern never repeats — it looks organic rather than mechanical.
#
# Styles:
#   welcome  — big, energetic whole-body engagement
#   talk     — medium expressive nods + antenna flutter
#   curious  — extra roll (ear-to-shoulder tilt), slower, questioning

STYLES = {
    "welcome": dict(
        ph=(0.14, 0.50), ph2=(0.05, 1.31),
        ya=(0.22, 0.31), ya2=(0.07, 0.73),
        ro=(0.07, 0.19), ro2=(0.03, 0.53),
        an=(0.55, 0.61), an2=(0.15, 1.17),
        by=(0.10, 0.17), by2=(0.03, 0.41),
    ),
    "talk": dict(
        ph=(0.10, 0.50), ph2=(0.04, 1.27),
        ya=(0.18, 0.27), ya2=(0.05, 0.71),
        ro=(0.05, 0.15), ro2=(0.02, 0.43),
        an=(0.42, 0.55), an2=(0.12, 1.09),
        by=(0.07, 0.13), by2=(0.02, 0.37),
    ),
    "curious": dict(
        ph=(0.08, 0.35), ph2=(0.03, 0.89),
        ya=(0.20, 0.21), ya2=(0.06, 0.59),
        ro=(0.12, 0.29), ro2=(0.04, 0.67),
        an=(0.32, 0.43), an2=(0.10, 0.97),
        by=(0.05, 0.11), by2=(0.02, 0.31),
    ),
}


def _wave(c, key, t):
    a1, f1 = c[key]
    a2, f2 = c[key + "2"]
    return a1 * math.sin(2 * math.pi * f1 * t) + a2 * math.sin(2 * math.pi * f2 * t)


def speak_and_animate(mini, audio_path: str, audio_duration: float):
    """
    Play audio and animate simultaneously.

    Style transitions (proportional to audio length):
      0–30 %  → 'welcome'  (energetic opening)
      30–55 % → 'curious'  (the question)
      55–100% → 'talk'     (listing topics)
    """
    proc = subprocess.Popen(
        ["aplay", "-D", SPEAKER, "-q", audio_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    phase_cuts = [audio_duration * 0.30, audio_duration * 0.55]
    hz = 20
    t0 = time.time()

    while proc.poll() is None:
        t = time.time() - t0
        if t < phase_cuts[0]:
            style = "welcome"
        elif t < phase_cuts[1]:
            style = "curious"
        else:
            style = "talk"

        c = STYLES[style]
        mini.set_target(
            head=create_head_pose(
                pitch=_wave(c, "ph", t),
                yaw=_wave(c, "ya", t),
                roll=_wave(c, "ro", t),
                degrees=False,
            ),
            antennas=[_wave(c, "an", t), -_wave(c, "an", t)],
            body_yaw=_wave(c, "by", t),
        )
        time.sleep(1.0 / hz)

    proc.wait()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Network School Greeter")

    print("  Generating speech...")
    audio_duration, audio_path = synth(GREETING)
    print(f"  Audio: {audio_duration:.1f} s")

    print("  >>> RECORD CUE: 3 beeps — hit record now! <<<")
    record_cue()
    time.sleep(1.0)   # 1 s after last beep before anything moves

    print("  Boot sequence...")
    boot_sequence()
    time.sleep(0.15)
    ready_blip()
    print("  Starting in 3 s...")
    time.sleep(3.0)

    print("  Starting daemon...")
    daemon_proc = start_daemon()

    try:
        emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")

        with ReachyMini(connection_mode="localhost_only",
                        media_backend="no_media",
                        spawn_daemon=False) as mini:
            mini.wake_up()

            print("  → welcoming gestures")
            mini.play_move(emotions.get("welcoming1"), play_frequency=80.0, sound=False)
            time.sleep(0.2)
            mini.play_move(emotions.get("enthusiastic1"), play_frequency=80.0, sound=False)
            time.sleep(0.3)

            print("  → speaking + animating")
            speak_and_animate(mini, audio_path, audio_duration)
            time.sleep(0.3)

            print("  → attentive pose")
            mini.play_move(emotions.get("attentive1"), play_frequency=80.0, sound=False)
            time.sleep(0.4)

            mini.goto_target(
                head=create_head_pose(),
                antennas=[0.0, 0.0],
                duration=1.0,
                body_yaw=0.0,
            )
            time.sleep(1.1)
            mini.goto_sleep()
            print("  Done.")

    finally:
        Path(audio_path).unlink(missing_ok=True)
        stop_daemon(daemon_proc)


if __name__ == "__main__":
    main()
