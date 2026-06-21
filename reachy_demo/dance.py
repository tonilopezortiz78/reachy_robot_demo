"""
reachy_demo/dance.py — Macarena beat-sync dance + sound effects.

Public API:
  DANCE_KEYWORDS  — set of trigger words in 10+ languages
  do_macarena(mini, dances, emotions, anim, log=None, funny_text=None,
              music_duration=15.0, injected_phrase=None, injected_at_s=6.0)
      Beat-synced Macarena with music, then silent dancing, then reaction.
      See do_macarena docstring for full flow.
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


def _dance_n_beats(mini, n, scale=1.6):
    """Dance `n` consecutive Macarena poses at beat rate. Returns True if stopped early."""
    for i in range(n):
        pose = _MACARENA_POSES[i % len(_MACARENA_POSES)]
        _beat(mini, pose, scale, time.time() + _BEAT)
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def do_macarena(mini, dances, emotions, anim, log=None, funny_text=None,
                music_duration=15.0, injected_phrase=None, injected_at_s=6.0):
    """
    Beat-synced Macarena show (~18 s + reaction):

      excited_chirp → music starts WITH injected TTS mixed in at injected_at_s
      → 3 escalating cycles + jump transitions → music stops at `music_duration`
      → 3 s silent dancing → confused head turns → speaks `funny_text`

    Since the routine and music are always the same, timing is hardcoded from
    the pre-analysed BPM. No audio detection needed — ffmpeg -t <duration>
    guarantees precise cutoff.

    Pauses the Animator for the full duration so beat-sync goto_target calls
    have sole servo control. Resumes in finally so it's always restored.
    """
    if log:
        log.event("  [dance] Macarena starting!")
    anim.pause()
    music_proc = None
    inj_wav = None
    try:
        excited_chirp()   # clears ALSA + signals excitement

        # ── Build ffmpeg command: music + optional injected TTS ──────
        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-stream_loop", "-1", "-i", str(MUSIC_PATH),
        ]
        if injected_phrase:
            inj_wav = synth_to_file(injected_phrase)
            ffmpeg_cmd += ["-i", inj_wav]
            delay_ms = int(injected_at_s * 1000)
            ffmpeg_cmd += [
                "-filter_complex",
                f"[0:a]volume=2.0[music];"
                f"[1:a]adelay={delay_ms}[inj];"
                f"[music][inj]amix=inputs=2:duration=first",
            ]
        else:
            ffmpeg_cmd += ["-af", "volume=2.0"]

        ffmpeg_cmd += ["-t", str(music_duration), "-f", "alsa", SPEAKER]

        music_proc = subprocess.Popen(ffmpeg_cmd)
        music_t0 = time.time()

        # ── Entry spins (0.48 s total) ─────────────────────────────
        _spin(mini,  1.4, dur=0.17)
        _spin(mini, -1.4, dur=0.17)
        _spin(mini,  0.0, dur=0.14)

        # ── Snap to beat boundary ─────────────────────────────────
        elapsed  = time.time() - music_t0
        beat_offset = math.ceil(elapsed / _BEAT)
        wait_snap = music_t0 + beat_offset * _BEAT - time.time()
        if wait_snap > 0:
            time.sleep(wait_snap)

        # ── Beat-synced cycles for the full music duration ─────────
        beat_count = 0
        while time.time() - music_t0 < music_duration:
            cycle = beat_count // len(_MACARENA_POSES)
            i = beat_count % len(_MACARENA_POSES)
            scale = 1.0 + min(cycle, 2) * 0.30  # clamp at 1.6

            target_t = music_t0 + (beat_offset + beat_count) * _BEAT
            if target_t > music_t0 + music_duration:
                break

            _beat(mini, _MACARENA_POSES[i], scale, target_t)
            beat_count += 1

            # Jump transitions at end of cycles 2 and 3
            if i == len(_MACARENA_POSES) - 1:
                if cycle == 2:
                    _jump(mini)
                    mini.play_move(dances.get("groovy_sway_and_roll"),
                                   play_frequency=80.0, sound=False)
                elif cycle == 3:
                    _jump(mini)

        # ── Wait for ffmpeg to finish (audio fully flushed) ─────────
        music_proc.wait()
        music_proc = None

        # ── 1.5 more seconds of silent dancing ────────────────────────
        silent_end = time.time() + 1.0
        i = beat_count % len(_MACARENA_POSES)
        while time.time() < silent_end:
            _beat(mini, _MACARENA_POSES[i % len(_MACARENA_POSES)], 1.6,
                  time.time() + _BEAT)
            i += 1

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

    finally:
        if music_proc is not None:
            music_proc.kill()
            music_proc.wait()
        if inj_wav:
            Path(inj_wav).unlink(missing_ok=True)
        mini.goto_target(head=create_head_pose(), antennas=[0.0, 0.0],
                         body_yaw=0.0, duration=0.5)
        time.sleep(0.6)
        anim.resume()
