"""
reachy_demo/dance.py — Macarena beat-sync dance + sound effects.

Public API:
  DANCE_KEYWORDS  — set of trigger words in 10+ languages
  do_macarena(mini, dances, emotions, anim, log=None, funny_text=None)
      Full ~20 s beat-synced Macarena show with music, jump transitions,
      and climax. Music stops → confused moves → speaks `funny_text` →
      returns to neutral. Pauses the Animator for sole servo control.
  excited_chirp()
      Two ascending frequency sweeps — used before/during dance to signal
      excitement AND to clear the ALSA device after TTS aplay exits.

Beat timing: 103.4 BPM, pre-analysed with librosa (BEAT = 0.5805 s).
Music:       ROOT/music/macarena.mp3 at +6 dB via ffmpeg → ALSA.
"""
import math
import subprocess
import time
from pathlib import Path

from reachy_demo.audio import SPEAKER
from reachy_demo.tts_edge import synth_to_file

ROOT = Path(__file__).parent.parent

# ── Trigger keywords (multilingual) ──────────────────────────────────────────

DANCE_KEYWORDS = {
    "dance", "dancing", "groove", "boogie", "moves", "move", "move it", "macarena",
    "bailar", "baila", "baile", "bailemos", "bailas",                  # Spanish
    "danser", "danse", "dansez",                                        # French
    "tanzen", "tanz",                                                   # German
    "ballare", "balla", "ballo",                                        # Italian
    "танцуй", "танцевать", "танец",                                     # Russian
    "踊", "踊れ", "ダンス", "おどって",                                # Japanese
    "跳舞", "舞",                                                       # Chinese
    "رقص", "ارقص",                                                     # Arabic
    "nac", "naach",                                                     # Hindi
}

# ── Sound effects ─────────────────────────────────────────────────────────────

def _chirp(f0, f1, dur, vol=0.65):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi",
         "-i", f"aevalsrc=sin(2*PI*({f0}*t+({f1}-{f0})*t*t/(2*{dur})))*{vol}:c=mono:s=22050",
         "-t", str(dur), "-f", "alsa", SPEAKER],
        check=False, stderr=subprocess.DEVNULL,
    )


def excited_chirp():
    """Two ascending sweeps.

    Call before starting the music ffmpeg: the blocking chirp clears the ALSA
    device after TTS aplay exits, preventing "Device or resource busy" when
    the music process tries to open the same device immediately after.
    """
    _chirp(500, 1800, 0.14, vol=0.75)
    time.sleep(0.04)
    _chirp(800, 2200, 0.12, vol=0.85)


# ── Macarena beat-sync constants ─────────────────────────────────────────────

MUSIC_PATH = ROOT / "music" / "macarena.mp3"
_BEAT = 0.5805   # 103.4 BPM, pre-analysed with librosa

# 8-pose arm cycle — one pose per beat.
# Fields: (pitch, yaw, roll, body_yaw, [ant_left, ant_right])
_MACARENA_POSES = [
    ( 0.08, -0.42,  0.10,   0.55, [ 0.10, -0.72]),  # right arm out
    ( 0.15, -0.52,  0.14,   0.80, [ 0.05, -0.85]),  # right arm high
    ( 0.08,  0.42, -0.10,  -0.55, [ 0.72, -0.10]),  # left arm out
    ( 0.15,  0.52, -0.14,  -0.80, [ 0.85, -0.05]),  # left arm high
    ( 0.04, -0.20,  0.30,   1.00, [ 0.60, -0.60]),  # right shoulder cross
    ( 0.04,  0.20, -0.30,  -1.00, [-0.60,  0.60]),  # left shoulder cross
    (-0.22,  0.05,  0.14,   1.30, [ 0.80,  0.80]),  # shimmy right
    (-0.14,  0.05, -0.14,  -1.40, [ 0.80,  0.80]),  # shimmy left
]

# ── Internal helpers ──────────────────────────────────────────────────────────

from reachy_mini.utils import create_head_pose   # noqa: E402 (after stdlib)


def _clamp(v, lim):
    return max(-lim, min(lim, v))


def _beat(mini, pose, scale, target_t):
    """Move to pose, sleeping until target_t (wall-clock drift correction)."""
    p, y, r, by, ants = pose
    dur = max(0.12, target_t - time.time() - 0.04)
    mini.goto_target(
        head=create_head_pose(
            pitch=_clamp(p * scale, 0.36),
            yaw=_clamp(y * scale, 1.50),
            roll=_clamp(r * scale, 0.36),
            degrees=False,
        ),
        antennas=[_clamp(ants[0] * scale, 0.80),
                  _clamp(ants[1] * scale, 0.80)],
        body_yaw=_clamp(by * scale, 1.40),
        duration=dur,
    )
    rem = target_t - time.time()
    if rem > 0:
        time.sleep(rem)


def _spin(mini, angle, dur=0.42):
    """Spin body with head looking in the spin direction."""
    mini.goto_target(
        head=create_head_pose(yaw=_clamp(angle * 0.25, 1.50), degrees=False),
        antennas=[0.0, 0.0],
        body_yaw=angle, duration=dur,
    )
    time.sleep(dur + 0.05)


def _spin360(mini):
    """360° illusion: blast to ±160°, snap across the gap, return to 0.
    Head tracks each spin direction for extra flair."""
    mini.goto_target(head=create_head_pose(pitch=0.10, yaw=0.50, degrees=False),
                     antennas=[0.80, -0.80], body_yaw=2.79, duration=0.22)
    time.sleep(0.02)
    mini.goto_target(head=create_head_pose(pitch=0.10, yaw=-0.50, degrees=False),
                     antennas=[-0.80, 0.80], body_yaw=-2.79, duration=0.18)
    time.sleep(0.02)
    mini.goto_target(head=create_head_pose(pitch=0.25, yaw=0.0, degrees=False),
                     antennas=[0.80, 0.80], body_yaw=0.0, duration=0.28)
    time.sleep(0.10)


def _jump(mini):
    """Slow push-down → instant snap-up (slingshot effect)."""
    mini.goto_target(head=create_head_pose(pitch=-0.38, roll=0.10, degrees=False),
                     antennas=[-0.50, -0.50], body_yaw=0.0, duration=0.50)
    time.sleep(0.02)
    mini.goto_target(head=create_head_pose(pitch=0.40, roll=-0.06, degrees=False),
                     antennas=[0.90, 0.90], body_yaw=0.0, duration=0.07)
    time.sleep(0.12)


# ── Public API ────────────────────────────────────────────────────────────────

def do_macarena(mini, dances, emotions, anim, log=None, funny_text=None):
    """
    Full beat-synced Macarena show (~30 s):
      excited_chirp → music starts → entry spins (double-speed, head tracking)
      → 3 escalating cycles (scale 1.0→1.3→1.6) + jump transitions
      → climax (3× spin360 + dizzy_spin + polyrhythm_combo + enthusiastic2
      + success1) → 10 s extra Macarena cycles → music stops
      → robot looks confused → speaks `funny_text`

    Pauses the Animator for the full duration so beat-sync goto_target calls
    have sole servo control. Resumes in finally so it's always restored.

    `funny_text` is spoken via edge-tts (synth_to_file) right after the
    confused moves, with a tiny head-bob during playback. Requires internet.
    """
    if log:
        log.event("  [dance] Macarena starting!")
    anim.pause()
    music_proc = None
    try:
        excited_chirp()   # clears ALSA + signals excitement

        music_proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-stream_loop", "-1", "-i", str(MUSIC_PATH),
             "-af", "volume=2.0", "-f", "alsa", SPEAKER],
        )
        music_t0 = time.time()

        # Entry spins — double speed with head tracking
        _spin(mini,  1.4, dur=0.17)
        _spin(mini, -1.4, dur=0.17)
        _spin(mini,  0.0, dur=0.14)

        # Snap to next clean beat boundary
        elapsed  = time.time() - music_t0
        beat_idx = math.ceil(elapsed / _BEAT)
        wait_snap = music_t0 + beat_idx * _BEAT - time.time()
        if wait_snap > 0:
            time.sleep(wait_snap)

        # 3 escalating cycles
        for cycle in range(3):
            scale = 1.0 + cycle * 0.30
            for i, pose in enumerate(_MACARENA_POSES):
                _beat(mini, pose, scale,
                      music_t0 + (beat_idx + cycle * len(_MACARENA_POSES) + i) * _BEAT)
            if cycle == 1:
                _jump(mini)
                mini.play_move(dances.get("groovy_sway_and_roll"),
                               play_frequency=80.0, sound=False)
            elif cycle > 1:
                _jump(mini)

        # Climax — no excited_chirp() here: music ffmpeg holds ALSA and chirps would fail
        _spin360(mini)
        mini.play_move(dances.get("dizzy_spin"),       play_frequency=80.0, sound=False)
        _spin360(mini)
        mini.play_move(dances.get("polyrhythm_combo"), play_frequency=80.0, sound=False)
        _spin360(mini)
        mini.play_move(emotions.get("enthusiastic2"),  play_frequency=80.0, sound=False)
        mini.play_move(emotions.get("success1"),       play_frequency=80.0, sound=False)

        # ── Extra 10 s: keep dancing while music plays ──────────────
        extra_end = time.time() + 10
        while time.time() < extra_end:
            for pose in _MACARENA_POSES:
                if time.time() >= extra_end:
                    break
                _beat(mini, pose, scale=1.6, target_t=time.time() + _BEAT)

    finally:
        # ── Stop music ──────────────────────────────────────────────
        if music_proc is not None:
            music_proc.terminate()
            try:
                music_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                music_proc.kill()
                music_proc.wait()

        # ── Confused — "where'd the music go?" (fast double-take) ───
        mini.goto_target(head=create_head_pose(pitch=-0.10, roll=0.15, degrees=False),
                         antennas=[-0.30, 0.50], body_yaw=0.30, duration=0.35)
        time.sleep(0.40)
        mini.goto_target(head=create_head_pose(pitch=-0.10, roll=-0.15, degrees=False),
                         antennas=[0.50, -0.30], body_yaw=-0.30, duration=0.35)
        time.sleep(0.40)

        # ── Speak the funny line while holding a slight bob ─────────
        if funny_text:
            wav = synth_to_file(funny_text)
            if wav:
                play_proc = subprocess.Popen(
                    ["aplay", "-D", SPEAKER, "-q", wav],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                # Small head-bob during TTS playback
                while play_proc.poll() is None:
                    mini.goto_target(
                        head=create_head_pose(pitch=0.05, yaw=0.10, degrees=False),
                        antennas=[0.20, 0.20], body_yaw=0.0, duration=0.25)
                    time.sleep(0.30)
                    mini.goto_target(
                        head=create_head_pose(pitch=0.05, yaw=-0.10, degrees=False),
                        antennas=[0.20, 0.20], body_yaw=0.0, duration=0.25)
                    time.sleep(0.30)
                play_proc.wait()
                Path(wav).unlink(missing_ok=True)

        # ── Return to neutral ───────────────────────────────────────
        mini.goto_target(head=create_head_pose(), antennas=[0.0, 0.0],
                         body_yaw=0.0, duration=0.5)
        time.sleep(0.6)
        anim.resume()
