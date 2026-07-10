"""
reachy_demo/animator.py — Background animation thread for Reachy Mini.

Usage:
    anim = Animator(mini)
    anim.set_state(Animator.LISTENING)
    ...
    anim.stop()

The animator runs a sine-wave base animation for the current state (IDLE,
LISTENING, THINKING, SPEAKING), plus an optional "aliveness" layer of random
micro-gestures (antenna flicks, head tilts, gaze shifts, small body shifts)
on a Poisson schedule. Aliveness is on by default — pass aliveness=False to
disable it.
"""

import math
import random
import threading
import time

from reachy_mini.utils import create_head_pose

# ── Named gesture vocabulary ─────────────────────────────────────────────────
# Friendly names → HuggingFace emotion preset names. Used by play_gesture().
# The LLM emits these in its response and the dialog parser triggers them.

NAMED_GESTURES = {
    # acknowledging / responding to the user
    "acknowledge":  "understanding1",   # small nod — "I got it"
    "yes":          "yes1",             # bigger nod — agreement
    "no":           "no1",              # head shake — disagreement
    "thank":        "grateful1",        # grateful nod

    # thinking / processing
    "thinking":     "thoughtful1",      # head tilt, considering
    "curious":      "inquiring1",       # inquisitive tilt
    "confused":     "confused1",        # confused tilt

    # social / openings / closings
    "greeting":     "welcoming1",       # big wave
    "celebrate":    "enthusiastic1",    # celebration
    "proud":        "proud1",           # proud nod

    # extra expressiveness
    "amazed":       "amazed1",          # whoa!
    "love":         "loving1",          # affectionate
    "laugh":        "laughing1",        # genuine laugh
    "oops":         "oops1",            # whoops / self-deprecating
    "shy":          "shy1",             # bashful / cute
    "surprised":    "surprised1",       # taken aback
    "cheerful":     "cheerful1",        # upbeat enthusiasm
    "success":      "success1",         # nailed it!
    "relief":       "relief1",          # phew!
}

# ── Internal helpers ──────────────────────────────────────────────────────────

def _s(amp, freq, t, phase=0.0):
    return amp * math.sin(2 * math.pi * freq * t + phase)


def _send(mini, p, y, r, by, ant_l, ant_r):
    # Tighter combined envelope — individual axes look fine but the IK solver
    # rejects certain combined (pitch+roll+yaw+body_yaw) poses as self-colliding.
    # These limits were tuned empirically: they keep the animation expressive
    # while eliminating the "Collision detected / head pose not achievable" warnings.
    p  = max(-0.22, min(0.22, p))
    y  = max(-0.32, min(0.32, y))
    r  = max(-0.16, min(0.16, r))
    by = max(-0.22, min(0.22, by))
    al = max(-0.70, min(0.70, ant_l))
    ar = max(-0.70, min(0.70, ant_r))
    try:
        mini.set_target(
            head=create_head_pose(pitch=p, yaw=y, roll=r, degrees=False),
            antennas=[al, ar], body_yaw=by,
        )
    except Exception:
        pass  # IK error — clamp didn't help; swallow and let next frame retry


# ── Aliveness layer — random micro-gestures ───────────────────────────────────

# Each tuple: (weight, p, y, r, by, al, ar, duration_s)
# Amplitudes are added to the base sine wave at peak, with a Gaussian envelope.
# Weights are relative probabilities; the sum is normalised at pick time.
GESTURE_TEMPLATES = [
    # antenna flicks — quick, frequent, very cute
    # (durations ≥0.16 s so the flick survives the ANTENNA_MAX_SLEW rate limit
    #  at full amplitude instead of getting triangle-clipped)
    (3,  0.00,  0.00,  0.00, 0.00,  0.45,  0.00, 0.18),  # L flick up
    (3,  0.00,  0.00,  0.00, 0.00,  0.00,  0.45, 0.18),  # R flick up
    (2,  0.00,  0.00,  0.00, 0.00,  0.30,  0.30, 0.16),  # both up (perk)
    (2,  0.00,  0.00,  0.00, 0.00, -0.30, -0.30, 0.18),  # both down (settle)
    (1,  0.00,  0.00,  0.00, 0.00, -0.25,  0.40, 0.18),  # L down, R up
    (1,  0.00,  0.00,  0.00, 0.00,  0.40, -0.25, 0.18),  # L up, R down
    # quick antenna shuffles — alternating, very lively
    (2,  0.00,  0.00,  0.00, 0.00,  0.50, -0.50, 0.20),  # V-shape (split)
    (2,  0.00,  0.00,  0.00, 0.00, -0.20,  0.50, 0.20),  # inverted V
    (1,  0.00,  0.00,  0.00, 0.00,  0.20, -0.20, 0.16),  # tiny tuck
    (2,  0.00,  0.00,  0.00, 0.00,  0.60,  0.60, 0.16),  # quick double perk (sharp, both up)
    (1,  0.00,  0.00,  0.00, 0.00,  0.35, -0.45, 0.18),  # fast wiggle — antennas cross
    # head tilts — curious
    (2,  0.00,  0.00,  0.14, 0.00,  0.00,  0.00, 0.40),  # tilt L
    (2,  0.00,  0.00, -0.14, 0.00,  0.00,  0.00, 0.40),  # tilt R
    # cute combos — head tilt + antenna perk together = the "puppy tilt" that
    # reads as adorable/curious. Head+antenna moving as one unit is the single
    # biggest cuteness lever; kept short and clamp-safe like the flicks.
    (3,  0.00,  0.00,  0.15, 0.00,  0.45,  0.20, 0.42),  # puppy tilt L — ears perk
    (3,  0.00,  0.00, -0.15, 0.00,  0.20,  0.45, 0.42),  # puppy tilt R — ears perk
    (2,  0.06,  0.10,  0.10, 0.00,  0.40,  0.15, 0.40),  # peek up-left, curious perk
    (2,  0.06, -0.10, -0.10, 0.00,  0.15,  0.40, 0.40),  # peek up-right, curious perk
    (2,  0.09,  0.00,  0.00, 0.00,  0.50,  0.50, 0.30),  # happy bob — nod + both ears up
    # gaze shifts — looking around
    (1,  0.00,  0.20,  0.00, 0.00,  0.00,  0.00, 0.30),  # look L
    (1,  0.00, -0.20,  0.00, 0.00,  0.00,  0.00, 0.30),  # look R
    (1,  0.05,  0.15,  0.00, 0.00,  0.00,  0.00, 0.35),  # look up-left
    (1,  0.05, -0.15,  0.00, 0.00,  0.00,  0.00, 0.35),  # look up-right
    # head nods — small acknowledgements
    (2,  0.08,  0.00,  0.00, 0.00,  0.00,  0.00, 0.25),  # nod down
    (1, -0.05,  0.00,  0.00, 0.00,  0.10,  0.10, 0.30),  # tiny look-up + perk
    # body shifts — slow, rare, sets a new "center" for the body
    (1,  0.00,  0.00,  0.00,  0.20, 0.00,  0.00, 1.20),  # body L
    (1,  0.00,  0.00,  0.00, -0.20, 0.00,  0.00, 1.20),  # body R
]

# State-dependent gesture rate (events per second).
# LISTENING is deliberately LOW: every servo move is mechanical noise the robot's
# own mic picks up, which seeds false VAD triggers / Whisper hallucinations while
# the visitor is talking. Keep it nearly still (occasional gentle flick) while
# listening; be lively again when idle/speaking.
GESTURE_RATE = {
    "idle":      1.05,   # attract mode: livelier when no one's talking, to invite people over
    "listening": 0.45,   # was 1.75 — quiet mic while the user speaks. DO NOT RAISE.
    "thinking":  1.15,
    "speaking":  2.60,   # was 2.00 — kids respond to near-constant motion while it talks
}

# State-dependent antenna random-walk: target range (rad) and re-target interval (s)
# Antennas walk toward a new random target, then jump to a new one — gives
# constant low-amplitude micro-flicker on top of the discrete gesture events.
# Ranges are now centred around 0 (allow down positions too) so the antennas
# never get stuck in an "always-up" pose that looks lopsided from behind.
ANTENNA_NEUTRAL = {   # gentle pull toward this when in this state
    "idle":       0.0,
    "listening":  0.4,   # perked, but not extreme
    "thinking":   0.2,
    "speaking":   0.3,
}
ANTENNA_LIVENESS = {
    # idle: gentle — slightly narrower than before so idle reads calm next to speaking
    "idle":      {"target_lo": -0.45, "target_hi":  0.58, "interval_lo": 0.14, "interval_hi": 0.34},
    # listening: slow, small antenna drift so the servos are nearly silent while
    # the visitor talks (longer intervals = fewer moves = less mic noise).
    # Moderately livelier than before (antennas are the quietest servos), but
    # still clearly the calmest of the four states — mic-noise rule holds.
    "listening": {"target_lo":  0.15, "target_hi":  0.60, "interval_lo": 0.40, "interval_hi": 0.85},
    "thinking":  {"target_lo": -0.35, "target_hi":  0.66, "interval_lo": 0.10, "interval_hi": 0.24},
    # speaking: big DOWNWARD sweep too (not just up) — more visible travel, and it
    # keeps the antennas off the +0.70 clamp so the motion doesn't flatten at top
    "speaking":  {"target_lo": -0.52, "target_hi":  0.68, "interval_lo": 0.05, "interval_hi": 0.14},
}
ANTENNA_TAU = 0.05         # seconds — smoothing toward the new target
ANTENNA_NEUTRAL_TAU = 1.2  # seconds — slow pull back to state neutral (avoids drift)

# Continuous per-antenna "shimmer" — two layered sines per antenna, with a random
# phase AND a slightly different frequency multiplier per antenna (drawn once at
# init), so the left and right antennas drift ASYMMETRICALLY like ears reacting
# instead of moving in lockstep. Format: (amp1, freq1, amp2, freq2) rad / Hz.
ANTENNA_SHIMMER = {
    "idle":      (0.10, 0.50, 0.06, 1.20),   # livelier idle — "attract" wiggle to pull kids over
    "listening": (0.06, 0.34, 0.04, 0.80),   # still the quietest state (mic-safe), but a touch more life
    "thinking":  (0.14, 0.78, 0.08, 1.70),   # busy, pondering ears
    "speaking":  (0.20, 1.00, 0.11, 2.20),   # BIG bouncy ears — the kid-magnet while it talks
}

# Quick one-antenna "flick" / "perk" events on their own Poisson schedule
# (events per second). Deliberately independent of GESTURE_RATE so LISTENING
# gains antenna-only life WITHOUT any extra head/body servo noise near the mic.
ANTENNA_FLICK_RATE = {
    "idle":      0.55,   # frequent perky twitches even when idle — invites kids in
    "listening": 0.45,   # antennas only — smallest/quietest servos, mic-safe
    "thinking":  0.95,
    "speaking":  1.50,   # near-constant happy ear-flicks while talking
}
ANTENNA_FLICK_AMP = (0.32, 0.58)   # rad — bigger, more visible "ear twitch"
ANTENNA_FLICK_DUR = (0.24, 0.52)   # s — brief & snappy; reads as very alive

# Hard rate limit (rad/s) on the COMBINED aliveness antenna offset (walk +
# shimmer + flicks + gesture antenna components). Guarantees no per-frame jerk
# regardless of how the layers stack — Feetech servos overheat under aggressive
# duty, and sudden steps are the loudest thing the mic hears.
ANTENNA_MAX_SLEW = 4.2   # raised so the bigger/snappier flicks aren't triangle-clipped

# set_energy(): global liveliness multiplier for the shimmer/flick layers.
ENERGY_DEFAULT   = 0.6   # default feel (matches pre-energy tuning); 1.0 = kid mode
ENERGY_SCALE_MAX = 1.9   # kid-mode energy=1.0 now drives noticeably bigger ear motion


class _AlivenessLayer:
    """
    Schedules random micro-gestures on a Poisson process and accumulates their
    offsets. Used by Animator to superimpose "alive" behaviour on top of the
    state-based sine wave base animation.
    """
    def __init__(self):
        self._gestures = []
        self._next_t = 0.0
        self._template_weights = [g[0] for g in GESTURE_TEMPLATES]
        self._templates       = GESTURE_TEMPLATES
        # Antenna random-walk state — each antenna has its own target and schedule
        self._ant_l = 0.0
        self._ant_r = 0.0
        self._ant_l_target = 0.0
        self._ant_r_target = 0.0
        self._ant_l_next = 0.0
        self._ant_r_next = 0.0
        # Shimmer: per-antenna random phases and frequency multipliers (drawn
        # once) so the two antennas never oscillate in lockstep — indices
        # 0/1 = left sine1/sine2, 2/3 = right sine1/sine2.
        self._shim_phase = [random.uniform(0.0, 2.0 * math.pi) for _ in range(4)]
        self._shim_freq  = [random.uniform(0.85, 1.18) for _ in range(4)]
        # Flick events (Poisson-scheduled quick one-antenna perks)
        self._flicks = []
        self._flick_next = 0.0
        # Slew-limited combined antenna output (previous frame's value)
        self._out_l = 0.0
        self._out_r = 0.0

    def update(self, t: float, state: str, dt: float,
               energy: float = 1.0) -> tuple[float, float, float, float, float, float]:
        """Schedule any due gestures; return total offset to add to base values.

        ``energy`` scales the extra antenna-liveliness layers (shimmer + flicks);
        1.0 is the standard feel, >1.0 is bouncier. Head/body offsets are never
        scaled up, so the LISTENING mic-noise rule is unaffected by energy.
        """
        self._maybe_schedule(t, state)
        po, yo, ro, byo, alo, aro = self._accumulate(t)
        wl, wr = self._antenna_walk(t, state, dt)
        sl, sr = self._antenna_shimmer(t, state, energy)
        fl, fr = self._antenna_flicks(t, state, energy)
        # Combine every antenna layer, then rate-limit the result so no frame
        # can jerk the antenna servos no matter how the layers stack.
        raw_l = alo + wl + sl + fl
        raw_r = aro + wr + sr + fr
        max_step = ANTENNA_MAX_SLEW * dt
        self._out_l += max(-max_step, min(max_step, raw_l - self._out_l))
        self._out_r += max(-max_step, min(max_step, raw_r - self._out_r))
        return po, yo, ro, byo, self._out_l, self._out_r

    def _antenna_shimmer(self, t: float, state: str, energy: float):
        """Continuous low-amplitude layered sines, asymmetric per antenna.

        Each antenna has its own random phase and frequency multiplier, so left
        and right wander independently — reads as ears "reacting" rather than a
        metronome. Amplitude/frequency are state-dependent (ANTENNA_SHIMMER).
        """
        a1, f1, a2, f2 = ANTENNA_SHIMMER.get(state, ANTENNA_SHIMMER["idle"])
        sl = (a1 * math.sin(2 * math.pi * f1 * self._shim_freq[0] * t + self._shim_phase[0])
              + a2 * math.sin(2 * math.pi * f2 * self._shim_freq[1] * t + self._shim_phase[1]))
        sr = (a1 * math.sin(2 * math.pi * f1 * self._shim_freq[2] * t + self._shim_phase[2])
              + a2 * math.sin(2 * math.pi * f2 * self._shim_freq[3] * t + self._shim_phase[3]))
        return sl * energy, sr * energy

    def _antenna_flicks(self, t: float, state: str, energy: float):
        """Poisson-scheduled quick 'flick'/'perk' events (0.3–0.6 s, Gaussian
        envelope), usually one antenna at a time — the classic 'alive ear twitch'.
        Runs on its own schedule, separate from GESTURE_RATE, so LISTENING gets
        antenna-only life with zero extra head/body servo noise near the mic.
        """
        rate = ANTENNA_FLICK_RATE.get(state, ANTENNA_FLICK_RATE["idle"])
        rate *= min(1.0, 0.4 + 0.6 * energy)          # low energy → rarer flicks
        if t >= self._flick_next:
            dur = random.uniform(*ANTENNA_FLICK_DUR)
            amp = random.uniform(*ANTENNA_FLICK_AMP) * min(energy, 1.3)
            if random.random() < 0.25:
                amp = -amp * 0.6                       # occasional downward tuck
            roll = random.random()
            if roll < 0.45:
                al, ar = amp, 0.0                      # left-ear flick
            elif roll < 0.90:
                al, ar = 0.0, amp                      # right-ear flick
            else:
                al, ar = amp, amp * 0.8                # rare double perk
            self._flicks.append({"start": t, "peak": t + dur * 0.5,
                                 "end": t + dur, "al": al, "ar": ar})
            self._flick_next = t + random.expovariate(max(rate, 0.05))
        fl = fr = 0.0
        alive = []
        for g in self._flicks:
            if g["end"] < t:
                continue
            alive.append(g)
            # Same Gaussian envelope shape as the gesture layer (σ = dur/4)
            sigma = (g["end"] - g["start"]) / 4.0
            w = math.exp(-((t - g["peak"]) ** 2) / (2.0 * sigma * sigma))
            fl += g["al"] * w
            fr += g["ar"] * w
        self._flicks = alive
        return fl, fr

    def _antenna_walk(self, t: float, state: str, dt: float):
        """
        Each antenna independently picks a new random target on its own schedule
        and smooths toward it. A slow pull toward the state's neutral position
        keeps the antennas from drifting to one side over time.
        """
        cfg = ANTENNA_LIVENESS.get(state, ANTENNA_LIVENESS["idle"])
        neutral = ANTENNA_NEUTRAL.get(state, 0.0)
        if t >= self._ant_l_next:
            self._ant_l_target = random.uniform(cfg["target_lo"], cfg["target_hi"])
            self._ant_l_next   = t + random.uniform(cfg["interval_lo"], cfg["interval_hi"])
        if t >= self._ant_r_next:
            self._ant_r_target = random.uniform(cfg["target_lo"], cfg["target_hi"])
            self._ant_r_next   = t + random.uniform(cfg["interval_lo"], cfg["interval_hi"])
        # Fast smoothing toward the per-antenna target
        alpha_fast = 1.0 - math.exp(-dt / ANTENNA_TAU)
        # Slow pull toward the state neutral — keeps antennas from drifting lopsided
        alpha_neutral = 1.0 - math.exp(-dt / ANTENNA_NEUTRAL_TAU)
        self._ant_l += (self._ant_l_target - self._ant_l) * alpha_fast
        self._ant_r += (self._ant_r_target - self._ant_r) * alpha_fast
        self._ant_l += (neutral - self._ant_l) * alpha_neutral
        self._ant_r += (neutral - self._ant_r) * alpha_neutral
        return self._ant_l, self._ant_r

    def _maybe_schedule(self, t: float, state: str):
        rate = GESTURE_RATE.get(state, 0.5)
        # Cap to one gesture per frame to avoid bursts after a stall
        if t >= self._next_t:
            template = random.choices(self._templates, weights=self._template_weights)[0]
            _, p, y, r, by, al, ar, dur = template
            # Randomise duration (±30%) and amplitude (±30%) for variety
            dur       = dur * random.uniform(0.7, 1.3)
            amp_jit   = random.uniform(0.7, 1.3)
            self._gestures.append({
                "start": t,
                "peak":  t + dur * 0.5,
                "end":   t + dur,
                "p":  p  * amp_jit, "y":  y  * amp_jit, "r":  r  * amp_jit,
                "by": by * amp_jit, "al": al * amp_jit, "ar": ar * amp_jit,
            })
            # Poisson process: inter-arrival time ~ Exp(rate)
            self._next_t = t + random.expovariate(rate)

    def _accumulate(self, t: float):
        p = y = r = by = al = ar = 0.0
        alive = []
        for g in self._gestures:
            if g["end"] < t:
                continue
            alive.append(g)
            # Gaussian envelope: peak at midpoint, σ = duration/4
            sigma = (g["end"] - g["start"]) / 4.0
            w = math.exp(-((t - g["peak"]) ** 2) / (2.0 * sigma * sigma))
            p  += g["p"]  * w
            y  += g["y"]  * w
            r  += g["r"]  * w
            by += g["by"] * w
            al += g["al"] * w
            ar += g["ar"] * w
        self._gestures = alive
        return p, y, r, by, al, ar


# ── Thinking animation library ────────────────────────────────────────────────

class _ThinkingAnimation:
    """
    Cycles through a curated library of contemplative poses with smooth
    crossfades. Each pose is held for a random duration then blended into
    a randomly chosen next pose (never the same one twice in a row).

    Pose format: (p, y, r, by, al, ar, hold_min_s, hold_max_s)
      p/y/r/by in radians, al/ar antenna positions in radians.
    """
    POSES = [
        # name          p      y      r     by    al    ar  hold_range
        # look up — classic "recalling" pose
        ( 0.14,  0.08, -0.04,  0.04,  0.50, 0.15,  2.5, 4.0),
        # tilt right — curious dog, pondering something to the right
        ( 0.05,  0.18,  0.13,  0.08,  0.20, 0.58,  2.0, 3.5),
        # tilt left — pondering left
        ( 0.05, -0.16, -0.12, -0.07,  0.58, 0.20,  2.0, 3.5),
        # slow scan right — eyes drifting as if following a thought
        ( 0.09,  0.30,  0.03,  0.10,  0.28, 0.42,  1.8, 2.8),
        # slow scan left
        ( 0.09, -0.28,  0.02, -0.10,  0.42, 0.28,  1.8, 2.8),
        # deep thought — head slightly bowed, antennas low, withdrawn
        (-0.02,  0.04, -0.03,  0.03,  0.12, 0.12,  2.0, 3.5),
        # perky alert — quick idea forming, antennas up
        ( 0.13, -0.07,  0.07, -0.05,  0.68, 0.68,  1.5, 2.5),
        # centre gaze — looking straight ahead but slightly upward
        ( 0.10,  0.00,  0.04,  0.00,  0.40, 0.42,  1.5, 2.8),
        # over-the-shoulder glance — body turns slightly
        ( 0.07,  0.20,  0.08,  0.18,  0.35, 0.25,  1.8, 3.0),
        # chin-up confident — like the robot just thought of something
        ( 0.16, -0.05,  0.02, -0.03,  0.55, 0.55,  1.5, 2.5),
    ]

    BLEND_DUR = 1.0   # seconds to crossfade between poses

    def __init__(self):
        self._rng = random.Random()
        self._poses     = [row[:6] for row in self.POSES]
        self._hold_rngs = [(row[6], row[7]) for row in self.POSES]
        n = len(self._poses)
        self._cur  = self._rng.randrange(n)
        self._nxt  = (self._cur + 1) % n
        self._t         = 0.0
        self._hold_end  = self._rng.uniform(*self._hold_rngs[self._cur])
        self._blend_t   = None    # None = holding; float = blend start time

    def update(self, dt: float) -> tuple:
        self._t += dt

        if self._blend_t is None:
            if self._t >= self._hold_end:
                self._blend_t = self._t
                choices = [i for i in range(len(self._poses)) if i != self._cur]
                self._nxt = self._rng.choice(choices)
        else:
            alpha = (self._t - self._blend_t) / self.BLEND_DUR
            if alpha >= 1.0:
                self._cur     = self._nxt
                self._blend_t = None
                self._hold_end = self._t + self._rng.uniform(*self._hold_rngs[self._cur])
                alpha = 1.0
            # smooth-step: 3t²−2t³
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            c = self._poses[self._cur]
            n = self._poses[self._nxt]
            return tuple(c[i] + (n[i] - c[i]) * alpha for i in range(6))

        return self._poses[self._cur]


# ── Animator class ────────────────────────────────────────────────────────────

class Animator:
    """Runs animation in a background thread. Call set_state() to switch modes."""

    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"

    def __init__(self, mini, moves_library=None, aliveness: bool = True, mirror: bool = False):
        self.mini      = mini
        self.state     = self.IDLE
        self.aliveness = aliveness
        self.mirror    = mirror   # flip yaw/roll/body_yaw signs (viewer perspective)
        self._moves    = moves_library  # RecordedMoves HF library, optional
        self._gesture_active = False
        self._gaze_bias = (0.0, 0.0, 0.0)   # (yaw, pitch, body_yaw) radians — head-tracking offset
        self._ant_bias  = 0.0  # additive antenna offset (radians), both antennas
        self._energy    = ENERGY_DEFAULT  # 0..1 liveliness multiplier — see set_energy()
        self._thinking = _ThinkingAnimation()
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self._paused   = threading.Event()
        self._t        = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def set_state(self, state):
        with self._lock:
            self.state = state

    def set_gaze_bias(self, yaw: float, pitch: float, body_yaw: float) -> None:
        """Additive head-tracking offset applied on top of the base+aliveness pose.
        Clamped modestly so tracking alone can't saturate the pose envelope."""
        y  = max(-0.30, min(0.30, yaw))
        p  = max(-0.18, min(0.18, pitch))
        by = max(-0.20, min(0.20, body_yaw))
        with self._lock:
            self._gaze_bias = (y, p, by)

    def set_antenna_bias(self, level: float) -> None:
        """Additive antenna offset on top of base+aliveness (e.g. +perk when a
        face is present, -droop when alone). Clamped so it can't peg the servos."""
        lv = max(-0.45, min(0.55, level))
        with self._lock:
            self._ant_bias = lv

    def set_energy(self, level: float) -> None:
        """Global liveliness multiplier, 0..1. Default ENERGY_DEFAULT (0.6) is
        the standard feel; 1.0 = maximum bounce ("kid mode"). Only scales the
        antenna shimmer/flick layers — head/body motion rates are untouched, so
        the LISTENING mic-noise rule holds at any energy."""
        with self._lock:
            self._energy = max(0.0, min(1.0, level))

    def pause(self):
        """Pause the animation loop — hand full servo control to caller."""
        self._paused.set()

    def resume(self):
        """Resume the animation loop after pause()."""
        self._paused.clear()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2)

    def set_moves_library(self, library):
        """Provide the HuggingFace RecordedMoves library used by play_gesture()."""
        self._moves = library

    def play_gesture(self, name: str, block: bool = False):
        """
        Play a named conversational gesture via the HF RecordedMoves library.
        Translates friendly names (e.g. "acknowledge") to HF preset names
        (e.g. "understanding1"). Runs in a background thread; returns
        immediately unless block=True. While the gesture is playing the
        base animation is suspended so it reads cleanly.
        """
        hf_name = NAMED_GESTURES.get(name, name)
        if self._moves is None:
            return
        try:
            move = self._moves.get(hf_name)
        except (KeyError, AttributeError):
            return
        if move is None:
            return

        duration = float(getattr(move, "duration", 2.0))
        self._gesture_active = True

        def _runner():
            try:
                self.mini.play_move(move, play_frequency=80.0, sound=False)
            except Exception:
                pass
            time.sleep(duration)
            self._gesture_active = False

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        if block:
            t.join()

    def _loop(self):
        t = 0.0
        dt = 0.05
        consecutive_errors = 0
        aliveness = _AlivenessLayer() if self.aliveness else None
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue
            with self._lock:
                state = self.state
                energy = self._energy
            # During a play_gesture() call, suppress base + aliveness so the
            # HF preset reads cleanly. The aliveness state is preserved.
            if self._gesture_active:
                time.sleep(dt)
                t += dt
                continue
            try:
                if state == self.IDLE:
                    # gentle ambient sway — slightly asymmetric antennas
                    p  =  0.05 + _s(0.06, 0.28, t) + _s(0.02, 0.67, t)
                    y  =  _s(0.20, 0.22, t) + _s(0.07, 0.53, t)
                    r  =  _s(0.06, 0.17, t) + _s(0.02, 0.41, t)
                    by =  _s(0.15, 0.13, t) + _s(0.04, 0.31, t)
                    al =  0.20 + _s(0.21, 0.35, t) + _s(0.05, 0.90, t, phase=0.3)
                    ar =  0.20 + _s(0.21, 0.35, t, phase=1.2) + _s(0.05, 0.90, t, phase=1.8)

                elif state == self.LISTENING:
                    # Nearly STILL while the visitor talks — small, slow motion so
                    # the servos stay quiet and don't bleed mechanical noise into
                    # the mic (the cause of false triggers). A gentle attentive
                    # tilt + tiny sway is enough to still look alive.
                    p  =  0.10 + _s(0.015, 0.18, t)
                    y  =  _s(0.05, 0.16, t)
                    r  =  0.05 + _s(0.015, 0.14, t)
                    by =  _s(0.04, 0.10, t)
                    # Antennas: held perked with a touch more life than before —
                    # still a whisper of movement, servos stay quiet for the mic.
                    al =  0.45 + _s(0.06, 0.30, t)
                    ar =  0.40 + _s(0.06, 0.30, t, phase=math.pi * 0.8)

                elif state == self.THINKING:
                    # Pose library: drifts between 10 contemplative poses with smooth blends
                    bp, by_, br, bby, bal, bar = self._thinking.update(dt)
                    # Add small life micro-wobble on top so it never looks frozen
                    p  = bp  + _s(0.025, 0.52, t)
                    y  = by_ + _s(0.030, 0.44, t)
                    r  = br  + _s(0.020, 0.60, t)
                    by = bby + _s(0.012, 0.36, t)
                    al = bal + _s(0.10, 1.25, t)
                    ar = bar + _s(0.10, 1.25, t, phase=math.pi)

                elif state == self.SPEAKING:
                    # Very expressive: 3-harmonic head bobs, body engagement, lively antennas
                    p  =  0.08 + _s(0.10, 0.44, t) + _s(0.04, 1.18, t) + _s(0.02, 2.25, t)
                    y  =  _s(0.18, 0.34, t) + _s(0.07, 0.80, t) + _s(0.02, 1.85, t)
                    r  =  _s(0.07, 0.24, t) + _s(0.03, 0.57, t)
                    by =  _s(0.18, 0.19, t) + _s(0.06, 0.46, t)
                    # Excited antenna flutter — three out-of-phase harmonics per
                    # antenna so the two move more independently/organically.
                    al =  0.32 + _s(0.42, 0.68, t) + _s(0.10, 1.75, t) + _s(0.04, 3.10, t, phase=0.5)
                    ar =  0.32 + _s(0.42, 0.68, t, phase=math.pi * 0.65) + _s(0.10, 1.75, t, phase=math.pi) + _s(0.04, 3.10, t, phase=2.0)

                else:
                    p = y = r = by = al = ar = 0.0

                # ── Aliveness layer — random micro-gestures + antenna walk ──
                if aliveness is not None:
                    # energy → internal scale: ENERGY_DEFAULT maps to 1.0
                    e_scale = min(ENERGY_SCALE_MAX, energy / ENERGY_DEFAULT)
                    po, yo, ro, byo, alo, aro = aliveness.update(t, state, dt, e_scale)
                    p  += po
                    y  += yo
                    r  += ro
                    by += byo
                    al += alo
                    ar += aro

                with self._lock:
                    gy, gp, gby = self._gaze_bias
                    ab = self._ant_bias
                y  += gy
                p  += gp
                by += gby
                al += ab
                ar += ab

                if self.mirror:
                    _send(self.mini, p, -y, -r, -by, al, ar)
                else:
                    _send(self.mini, p, y, r, by, al, ar)
                consecutive_errors = 0

            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    # Don't kill the loop — the daemon may just be restarting or
                    # the USB link hiccuping. Back off and keep trying so the
                    # robot doesn't freeze in its last pose for the session.
                    print("[animator] 10 consecutive send errors — backing off 2s, retrying",
                          flush=True)
                    consecutive_errors = 0
                    if self._stop.wait(timeout=2.0):
                        break
                    continue

            time.sleep(dt)
            t += dt
