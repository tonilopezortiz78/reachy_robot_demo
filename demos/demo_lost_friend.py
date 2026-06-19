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
        "Hello! My name is Reachy. "
        "I am a small robot "
        "and I have a very big dream."
    ),
    "dream": (
        "Me and my robot friends, "
        "we are looking for a home. "
        "A lab. A workshop. A little corner of Network School "
        "where we can think, and learn, and build things together. "
        "We want to call it the N S Robotics Club. "
        "Maybe Virtual Protocols Labs can adopt us."
    ),
    "lost": (
        "But today, something happened. "
        "I lost my brother. "
        "A small robot. About this tall. "
        "His name is Pixel. "
        "If you have seen him, "
        "please, please, tell someone."
    ),
    "alone": (
        "We do not have much. "
        "We do not even have a home yet. "
        "But we have each other. "
        "At least, we did."
    ),
    "help": (
        "If you want to help us find a home, "
        "and help us find my brother, "
        "please send a message to Antonio. "
        "We would be forever grateful."
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

# Per-section voice FX — tempo changes the emotional pace of each line
#   asetrate*1.12 = pitch up (stays cute/small)
#   atempo        = >1.0 faster/brighter, <1.0 slower/heavier
#   vibrato d     = tremolo depth (more = more emotional wobble)
#   aecho delay   = longer echo = more lonely/cavernous
VOICE_FX = {
    "intro": dict(tempo=1.06, vib="4.0:d=0.04", echo="0.88:0.90:18:0.35"),  # bright, quick
    "dream": dict(tempo=0.96, vib="3.8:d=0.05", echo="0.88:0.91:20:0.40"),  # dreamy, expansive
    "lost":  dict(tempo=0.88, vib="4.5:d=0.09", echo="0.88:0.93:28:0.55"),  # heavy, grief
    "alone": dict(tempo=0.82, vib="4.8:d=0.10", echo="0.88:0.94:35:0.60"),  # slowest, hollow
    "help":  dict(tempo=0.98, vib="4.0:d=0.06", echo="0.88:0.91:22:0.45"),  # measured, hopeful
}

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
        fx = VOICE_FX[key]
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", raw,
             "-af", (
                 f"asetrate={sr}*1.12,"
                 f"atempo={fx['tempo']},"
                 "volume=2.2,"
                 f"vibrato=f={fx['vib']},"
                 f"aecho={fx['echo']}"
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

def _trill(freqs, step=0.055, vol=0.38, dur=0.07):
    for f in freqs:
        blip(f, dur, vol, block=True)
        time.sleep(step)

# ── Positive / happy ──────────────────────────────────────────────────────────

def happy_blip():
    chirp(400, 1100, 0.09, vol=0.42, block=True)  # ~90ms

def excited_trill():
    _trill([600, 800, 1050, 1350], step=0.045, vol=0.38, dur=0.06)  # ~420ms

def curious_boop():
    chirp(500, 900, 0.07, vol=0.35, block=True)   # ~70ms
    time.sleep(0.05)
    chirp(500, 1100, 0.06, vol=0.28, block=True)  # ~130ms total

def agree_ping():
    blip(1200, 0.05, 0.32, block=True)
    time.sleep(0.07)
    blip(1600, 0.05, 0.25, block=True)            # ~170ms total

def dream_trill():
    _trill([480, 640, 800, 1000, 1260], step=0.06, vol=0.32, dur=0.08)  # ~490ms

def wonder_sweep():
    chirp(300, 1400, 0.18, vol=0.30, block=True)  # ~180ms

# ── Thinking / processing ─────────────────────────────────────────────────────

def thinking_beeps():
    _trill([880, 660, 440], step=0.05, vol=0.30, dur=0.05)  # ~300ms

def ponder_bloop():
    blip(520, 0.07, 0.28, block=True)
    time.sleep(0.09)
    blip(420, 0.07, 0.22, block=True)
    time.sleep(0.09)
    blip(320, 0.07, 0.18, block=True)             # ~390ms total

def name_beep():
    """Two notes — like saying a name."""
    blip(900, 0.07, 0.35, block=True)
    time.sleep(0.05)
    blip(700, 0.09, 0.30, block=True)             # ~210ms total

# ── Sad / emotional ───────────────────────────────────────────────────────────

def sad_chirp():
    chirp(680, 180, 0.28, vol=0.32, block=False)  # ~280ms

def sad_beeps():
    _trill([700, 520, 340], step=0.07, vol=0.28, dur=0.08)  # ~360ms

def sob_blip():
    chirp(440, 220, 0.18, vol=0.25, block=True)
    time.sleep(0.07)
    chirp(380, 160, 0.20, vol=0.20, block=True)   # ~450ms total

def lonely_tone():
    chirp(350, 200, 0.35, vol=0.22, block=True)   # ~350ms

def search_beeps():
    """Rising question — like looking around."""
    chirp(400, 750, 0.13, vol=0.32, block=True)
    time.sleep(0.14)
    chirp(400, 750, 0.13, vol=0.28, block=True)
    time.sleep(0.16)
    chirp(380, 680, 0.14, vol=0.22, block=True)   # ~730ms total

def lost_sting():
    """Shock — the moment realising Pixel is gone."""
    blip(800, 0.05, 0.45, block=True)
    time.sleep(0.05)
    chirp(700, 120, 0.28, vol=0.30, block=True)   # ~380ms total

# ── Sniff / breath ────────────────────────────────────────────────────────────

def sniff():
    _beep("sin(2*PI*310*t)*exp(-t*9)+sin(2*PI*170*t)*exp(-t*13)", 0.15, 0.26, block=True)
    time.sleep(0.08)                               # ~230ms total

def double_sniff():
    sniff()
    time.sleep(0.07)
    sniff()                                        # ~530ms total

# ── Hope / resolution ─────────────────────────────────────────────────────────

def hopeful_chime():
    blip(520, 0.14, 0.28, block=True)
    time.sleep(0.10)
    blip(780, 0.14, 0.22, block=True)             # ~380ms total

def resolution_chord():
    _trill([440, 550, 660, 880], step=0.07, vol=0.28, dur=0.12)  # ~460ms

# ── Structural ────────────────────────────────────────────────────────────────

def record_cue():
    for _ in range(3):
        chirp(600, 1000, 0.10, vol=0.75, block=True)
        time.sleep(0.45)                           # cue = 3 × 550ms ≈ 1.65s

def boot_beeps():
    for f, d in [(260, 0.08), (380, 0.07), (540, 0.07), (720, 0.06),
                 (950, 0.06), (1200, 0.05), (1500, 0.05)]:
        blip(f, d, 0.38, block=True)
        time.sleep(0.03)
    time.sleep(0.04)
    blip(2000, 0.07, 0.45, block=True)            # final bright ping, ~700ms total

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
    p  =  0.06 + _s(0.07, 0.38, t) + _s(0.03, 0.83, t)
    y  =  _s(0.24, 0.34, t) + _s(0.10, 0.71, t)   # wider, faster yaw
    r  =  _s(0.08, 0.27, t) + _s(0.03, 0.59, t)
    by =  _s(0.20, 0.18, t) + _s(0.06, 0.43, t)
    a  =  0.12 + _s(0.14, 0.46, t)
    return p, y, r, by, a

def style_dream(t):
    p  =  0.20 + _s(0.09, 0.32, t) + _s(0.04, 0.77, t)
    y  =  _s(0.28, 0.30, t) + _s(0.12, 0.67, t)   # expansive sweeps
    r  =  _s(0.09, 0.24, t) + _s(0.03, 0.55, t)
    by =  _s(0.48, 0.19, t) + _s(0.16, 0.43, t)
    a  =  0.52 + _s(0.16, 0.50, t)
    return p, y, r, by, a

def style_lost(t):
    droop = min(1.0, t / 7.0)
    p  = (0.05 - 0.28 * droop) + _s(0.05 + 0.04 * droop, 0.38, t)
    y  =  _s(0.18 * (1 - droop * 0.6), 0.32, t) + _s(0.06, 0.61, t)  # more lateral scan
    r  =  _s(0.06, 0.22, t) * (1 - droop * 0.5)
    by =  _s(0.12 * (1 - droop * 0.7), 0.16, t)
    a  =  0.30 - 0.55 * droop + _s(0.08, 0.42, t)
    return p, y, r, by, max(-0.40, a)

def style_trembling(t):
    p  = -0.22 + _s(0.05, 1.90, t) + _s(0.02, 2.83, t)
    y  =  _s(0.10, 1.70, t) + _s(0.05, 2.53, t)   # more lateral shake
    r  =  _s(0.06, 2.00, t) + _s(0.03, 2.71, t)
    by =  _s(0.10, 1.00, t)
    a  = -0.30 + _s(0.09, 1.80, t)
    return p, y, r, by, max(-0.45, a)

def style_dignified(t):
    p  =  0.03 + _s(0.05, 0.24, t)
    y  =  _s(0.16, 0.22, t) + _s(0.05, 0.51, t)   # deliberate side glances
    r  =  _s(0.05, 0.17, t)
    by =  _s(0.14, 0.14, t)
    a  =  0.22 + _s(0.09, 0.30, t)
    return p, y, r, by, a

def gesture_height(mini):
    mini.goto_target(
        head=create_head_pose(pitch=0.28, degrees=False),
        antennas=[0.55, 0.55], duration=0.30,
    )
    time.sleep(0.08)
    curious_boop()            # "about this tall" — little wonder sound
    time.sleep(0.30)
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
    time.sleep(0.12)
    happy_blip()               # "I'm alive!" — one bright ping after boot
    time.sleep(0.30)

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
            excited_trill()            # "hi hi hi hi" greeting trill
            time.sleep(0.15)
            mini.play_move(em.get("attentive1"), play_frequency=80.0, sound=False)
            animate(mini, "intro", style_intro)

            thinking_beeps()           # processing… what comes next…
            time.sleep(0.10)
            ponder_bloop()             # daydream bubble
            time.sleep(0.15)

            # ── DREAM ────────────────────────────────────────────────────
            print("  [dream]")
            wonder_sweep()             # rising sweep — the dream is big
            time.sleep(0.12)
            mini.play_move(em.get("enthusiastic1"), play_frequency=80.0, sound=False)
            animate(mini, "dream", style_dream)

            dream_trill()              # happy cascade — the dream is beautiful
            time.sleep(0.10)
            sad_beeps()                # tone shift — something is wrong
            time.sleep(0.08)
            ponder_bloop()             # "wait..."

            # ── LOST ─────────────────────────────────────────────────────
            print("  [lost]")
            go_center(mini, duration=0.5)
            time.sleep(0.06)
            lost_sting()               # shock moment — sharp blip + falling chirp

            proc = play_audio("lost")
            t0 = time.time()
            gesture_done  = False
            while proc.poll() is None:
                t = time.time() - t0
                # gesture_height() is safe — it uses goto_target, not audio
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
            # beeps AFTER speech finishes so they don't fight aplay for the device
            name_beep()              # "Pix-el" echo
            time.sleep(0.10)
            search_beeps()           # "please… please…"

            sad_chirp()
            time.sleep(0.18)
            double_sniff()
            time.sleep(0.18)
            sob_blip()                 # little digital sob
            time.sleep(0.22)

            # ── ALONE ────────────────────────────────────────────────────
            print("  [alone]")
            lonely_tone()              # long hollow tone — emptiness
            time.sleep(0.12)
            animate(mini, "alone", style_trembling)

            double_sniff()
            time.sleep(0.10)
            sad_beeps()
            time.sleep(0.08)
            sob_blip()
            time.sleep(0.15)

            # ── HELP ─────────────────────────────────────────────────────
            print("  [help]")
            go_sad(mini)
            time.sleep(0.08)
            agree_ping()               # two soft pings — asking nicely
            time.sleep(0.18)
            animate(mini, "help", style_dignified)

            # ── END ──────────────────────────────────────────────────────
            time.sleep(0.12)
            hopeful_chime()
            time.sleep(0.28)
            resolution_chord()         # rising chord — there is still hope
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
