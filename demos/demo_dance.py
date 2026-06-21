"""
demo_dance.py — Network School Full Show (Macarena Edition)
===========================================================
Record cue → boot → greeting speech → beat-synced Macarena → climax → bow → sleep.

Beat-sync: pre-analyzed at 103.4 BPM (0.5805 s/beat).
Uses wall-clock drift correction — each pose snaps to the true beat boundary
so movements stay locked to the music even after SDK overhead accumulates.

Escalation: 3 cycles at scale 1.0 → 1.3 → 1.6 (amplitude grows each round).
Music volume: 2.0 (+6 dB vs default).

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

from reachy_demo.daemon import launch_daemon, wait_for_daemon, stop_daemon
from reachy_demo.tts_edge import synth_to_file  # PITCH +8Hz set in tts_edge.py

ROOT        = Path(__file__).parent.parent
SPEAKER     = "plughw:CARD=Audio,DEV=0"
CACHE_GREET = str(ROOT / "cache" / "dance_greeting.wav")
CACHE_TEASE = str(ROOT / "cache" / "dance_teaser.wav")

# ── Music ─────────────────────────────────────────────────────────────────────
# Swap this one line to use any MP3 from the music/ folder.
MUSIC = str(ROOT / "music" / "macarena.mp3")
# MUSIC = str(ROOT / "music" / "blipotron.mp3")
# ─────────────────────────────────────────────────────────────────────────────

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
    chirp(280, 800, 0.16); time.sleep(0.04)
    chirp(550, 1300, 0.14); time.sleep(0.03)
    chirp(900, 400, 0.18); time.sleep(0.04)
    chirp(700, 1600, 0.12); time.sleep(0.03)
    blip(1800, 0.05);       time.sleep(0.02)
    blip(2200, 0.06);       time.sleep(0.02)
    blip(2200, 0.05)

def excited_chirp():
    chirp(500, 1800, 0.14, vol=0.75)
    time.sleep(0.04)
    chirp(800, 2200, 0.12, vol=0.85)

def _wav_dur(path: str) -> float:
    with wave.open(path) as wf:
        return wf.getnframes() / wf.getframerate()

def _get_or_synth(text: str, cache_path: str) -> tuple[float, str]:
    """Return (duration, path). Generates via edge-tts once, then reuses cache."""
    if Path(cache_path).exists():
        return _wav_dur(cache_path), cache_path
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = synth_to_file(text)
    Path(tmp).rename(cache_path)
    return _wav_dur(cache_path), cache_path

# ---------------------------------------------------------------------------
# Greeting animation (layered sine waves — looks organic)
# ---------------------------------------------------------------------------

STYLES = {
    "welcome": dict(
        ph=(0.18, 0.50), ph2=(0.07, 1.31),
        ya=(0.28, 0.31), ya2=(0.10, 0.73),
        ro=(0.10, 0.19), ro2=(0.04, 0.53),
        an=(0.70, 0.61), an2=(0.22, 1.17),
        by=(0.85, 0.17), by2=(0.25, 0.41),
    ),
    "talk": dict(
        ph=(0.14, 0.50), ph2=(0.06, 1.27),
        ya=(0.24, 0.27), ya2=(0.08, 0.71),
        ro=(0.07, 0.15), ro2=(0.03, 0.43),
        an=(0.58, 0.55), an2=(0.18, 1.09),
        by=(0.75, 0.13), by2=(0.22, 0.37),
    ),
    "curious": dict(
        ph=(0.12, 0.35), ph2=(0.05, 0.89),
        ya=(0.26, 0.21), ya2=(0.09, 0.59),
        ro=(0.16, 0.29), ro2=(0.06, 0.67),
        an=(0.48, 0.43), an2=(0.15, 0.97),
        by=(0.65, 0.11), by2=(0.18, 0.31),
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
# Music — volume 2.0 (+6 dB louder than before)
# ---------------------------------------------------------------------------

def play_music(path: str) -> tuple[subprocess.Popen, float]:
    """Start music playback. Returns (process, wall-clock start time) for sync."""
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-stream_loop", "-1", "-i", path,
         "-af", "volume=2.0", "-f", "alsa", SPEAKER],
    )
    return proc, time.time()

# ---------------------------------------------------------------------------
# Beat-synced dance moves
# ---------------------------------------------------------------------------

# 8-pose Macarena cycle — one pose per beat (0.58 s).
# Mimics the iconic arm sequence: right out → right up → left out → left up →
# right cross → left cross → both-up shimmy-R → both-up shimmy-L.
# Antennas mirror the "arm" direction on each beat for maximum expressiveness.
# Scale factor grows each cycle so moves become bigger and bigger.
MACARENA_POSES = [
    # pitch   yaw     roll    body_yaw  [ant_L, ant_R]
    ( 0.08, -0.42,  0.10,   0.55, [ 0.10, -0.72]),  # 0 right arm out
    ( 0.15, -0.52,  0.14,   0.80, [ 0.05, -0.85]),  # 1 right arm high
    ( 0.08,  0.42, -0.10,  -0.55, [ 0.72, -0.10]),  # 2 left arm out
    ( 0.15,  0.52, -0.14,  -0.80, [ 0.85, -0.05]),  # 3 left arm high
    ( 0.04, -0.20,  0.30,   1.00, [ 0.60, -0.60]),  # 4 right shoulder cross
    ( 0.04,  0.20, -0.30,  -1.00, [-0.60,  0.60]),  # 5 left shoulder cross
    (-0.22,  0.05,  0.14,   1.30, [ 0.80,  0.80]),  # 6 shimmy right (antennas up!)
    (-0.14,  0.05, -0.14,  -1.40, [ 0.80,  0.80]),  # 7 shimmy left  (antennas up!)
]


def _clamp(v, lim):
    return max(-lim, min(lim, v))


def macarena_beat(mini, pose, scale: float, target_t: float):
    """
    Move to pose finishing just before target_t, then sleep to exact beat boundary.
    Wall-clock drift correction: duration is calculated from remaining time,
    so any SDK overhead in previous beats is automatically absorbed.
    """
    p, y, r, by, ants = pose
    now = time.time()
    move_dur = max(0.12, target_t - now - 0.04)  # 40 ms settling gap

    mini.goto_target(
        head=create_head_pose(
            pitch=_clamp(p * scale, 0.36),
            yaw=_clamp(y * scale, 1.50),
            roll=_clamp(r * scale, 0.36),
            degrees=False,
        ),
        antennas=[
            _clamp(ants[0] * scale, 0.80),
            _clamp(ants[1] * scale, 0.80),
        ],
        body_yaw=_clamp(by * scale, 1.40),
        duration=move_dur,
    )
    # Drift correction: sleep exactly until the beat
    remaining = target_t - time.time()
    if remaining > 0:
        time.sleep(remaining)


def jump(mini):
    """Slow push-down → instant snap-up (slingshot effect)."""
    print("    ↓ JUMP ↓")
    mini.goto_target(
        head=create_head_pose(pitch=-0.38, roll=0.10, degrees=False),
        antennas=[-0.50, -0.50],
        body_yaw=0.0, duration=0.50,
    )
    time.sleep(0.02)
    mini.goto_target(
        head=create_head_pose(pitch=0.40, roll=-0.06, degrees=False),
        antennas=[0.90, 0.90],
        body_yaw=0.0, duration=0.07,
    )
    time.sleep(0.12)


def macarena_section(mini, em, da, cycles: int = 3, music_t0: float = 0.0, beat_idx: int = 0):
    """
    Beat-synced Macarena cycles with per-beat drift correction.

    music_t0:  wall-clock time when music started (from play_music)
    beat_idx:  which beat index we're starting on (accounts for intro spins)

    Scale escalation: 1.0 → 1.30 → 1.60 — movements grow each round.
    Inter-cycle transitions escalate too: nothing → jump+groove → jump.
    """
    for c in range(cycles):
        scale = 1.0 + c * 0.30   # cycle 0: 1.0 | cycle 1: 1.30 | cycle 2: 1.60
        print(f"  -- Macarena cycle {c+1}/3 (scale ×{scale:.2f}) --")
        for i, pose in enumerate(MACARENA_POSES):
            beat_num = beat_idx + c * len(MACARENA_POSES) + i
            target_t = music_t0 + beat_num * BEAT
            macarena_beat(mini, pose, scale, target_t)
        if c == 1:
            # Transition cycle 2 → 3: jump + groove flash to build energy
            jump(mini)
            mini.play_move(da.get("groovy_sway_and_roll"), play_frequency=80.0, sound=False)
        elif c > 1:
            # After last cycle: jump to clear pose before climax
            jump(mini)


def spin(mini, angle, duration=0.45):
    """Hard body-turn to angle (radians) — visible dramatic transition."""
    mini.goto_target(
        head=create_head_pose(),
        antennas=[0.0, 0.0],
        body_yaw=angle, duration=duration,
    )
    time.sleep(duration + 0.05)


def spin360(mini):
    """
    360° spin illusion. Body yaw maxes at ±160° (±2.79 rad), so true 360 is
    impossible in one move. Instead: blast to +160° → instantly snap to -160°
    (the motor crosses the gap unseen) → return to 0. Looks like a full spin.
    Antennas fly out for drama.
    """
    # Phase 1: fast blast to max right with antennas spread
    mini.goto_target(
        head=create_head_pose(pitch=0.10, degrees=False),
        antennas=[0.80, -0.80],
        body_yaw=2.79, duration=0.22,
    )
    time.sleep(0.02)
    # Phase 2: instant snap to max left (motor crosses the unseen gap)
    mini.goto_target(
        head=create_head_pose(pitch=0.10, degrees=False),
        antennas=[-0.80, 0.80],
        body_yaw=-2.79, duration=0.18,
    )
    time.sleep(0.02)
    # Phase 3: return to center with antennas up — triumphant landing
    mini.goto_target(
        head=create_head_pose(pitch=0.25, degrees=False),
        antennas=[0.80, 0.80],
        body_yaw=0.0, duration=0.28,
    )
    time.sleep(0.10)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Network School — Macarena Show")

    print("  Loading speech...")
    greet_dur, WAV_GREET = _get_or_synth(GREETING, CACHE_GREET)
    tease_dur, WAV_TEASE = _get_or_synth(TEASER,   CACHE_TEASE)
    print(f"  Greeting: {greet_dur:.1f}s   Teaser: {tease_dur:.1f}s")

    daemon_proc = launch_daemon()

    print("\n  >>> RECORD CUE — hit record! <<<")
    record_cue()

    print("  Boot sequence...")
    boot_sequence()

    wait_for_daemon(daemon_proc)

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
            excited_chirp()
            speak_and_animate(mini, WAV_TEASE, tease_dur)
            excited_chirp()
            # Tease the dance before music: one groove flash to set expectations
            mini.play_move(da.get("chin_lead"), play_frequency=80.0, sound=False)
            excited_chirp()

            # ── Act 2: Macarena ──────────────────────────────────────────
            print("\n  ── Act 2: Macarena ──")
            beat_proc, music_t0 = play_music(MUSIC)
            try:
                # Dramatic entry spins
                spin(mini,  1.4, duration=0.35)
                spin(mini, -1.4, duration=0.35)
                spin(mini,  0.0, duration=0.28)

                # Snap to next clean beat boundary so cycles start in sync
                elapsed   = time.time() - music_t0
                beat_idx  = math.ceil(elapsed / BEAT)
                wait_snap = music_t0 + beat_idx * BEAT - time.time()
                if wait_snap > 0:
                    time.sleep(wait_snap)
                print(f"  Snapped to beat {beat_idx} ({elapsed:.2f}s into music)")

                # 3 escalating Macarena cycles (scale 1.0 → 1.30 → 1.60)
                macarena_section(mini, em, da, cycles=3, music_t0=music_t0, beat_idx=beat_idx)

                # Climax — punchy sequence using the dances library (each ~1.85s)
                print("  *** CLIMAX ***")
                excited_chirp()
                spin360(mini)
                mini.play_move(da.get("dizzy_spin"),           play_frequency=80.0, sound=False)
                spin360(mini)
                mini.play_move(da.get("polyrhythm_combo"),     play_frequency=80.0, sound=False)
                excited_chirp()
                spin360(mini)
                mini.play_move(em.get("enthusiastic2"),        play_frequency=80.0, sound=False)
                mini.play_move(em.get("success1"),             play_frequency=80.0, sound=False)

            finally:
                time.sleep(5)  # 5 more seconds of music before stopping
                beat_proc.terminate()
                beat_proc.wait()

            # ── Bow out ──────────────────────────────────────────────────
            print("\n  ── Bow out ──")
            spin(mini, 0.0, duration=0.4)
            mini.play_move(em.get("proud2"),  play_frequency=80.0, sound=False)
            time.sleep(0.2)
            mini.play_move(em.get("loving1"), play_frequency=80.0, sound=False)

            mini.goto_target(
                head=create_head_pose(), antennas=[0.0, 0.0],
                duration=1.0, body_yaw=0.0,
            )
            time.sleep(1.1)
            mini.goto_sleep()
            print("  Show complete!")

    finally:
        stop_daemon(daemon_proc)


if __name__ == "__main__":
    main()
