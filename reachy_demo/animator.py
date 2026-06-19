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
}

# ── Internal helpers ──────────────────────────────────────────────────────────

def _s(amp, freq, t, phase=0.0):
    return amp * math.sin(2 * math.pi * freq * t + phase)


def _send(mini, p, y, r, by, ant_l, ant_r):
    mini.set_target(
        head=create_head_pose(pitch=p, yaw=y, roll=r, degrees=False),
        antennas=[ant_l, ant_r], body_yaw=by,
    )


# ── Aliveness layer — random micro-gestures ───────────────────────────────────

# Each tuple: (weight, p, y, r, by, al, ar, duration_s)
# Amplitudes are added to the base sine wave at peak, with a Gaussian envelope.
# Weights are relative probabilities; the sum is normalised at pick time.
GESTURE_TEMPLATES = [
    # antenna flicks — quick, frequent, very cute
    (3,  0.00,  0.00,  0.00, 0.00,  0.45,  0.00, 0.15),  # L flick up
    (3,  0.00,  0.00,  0.00, 0.00,  0.00,  0.45, 0.15),  # R flick up
    (2,  0.00,  0.00,  0.00, 0.00,  0.30,  0.30, 0.12),  # both up (perk)
    (2,  0.00,  0.00,  0.00, 0.00, -0.30, -0.30, 0.15),  # both down (settle)
    (1,  0.00,  0.00,  0.00, 0.00, -0.25,  0.40, 0.18),  # L down, R up
    (1,  0.00,  0.00,  0.00, 0.00,  0.40, -0.25, 0.18),  # L up, R down
    # quick antenna shuffles — alternating, very lively
    (2,  0.00,  0.00,  0.00, 0.00,  0.50, -0.50, 0.20),  # V-shape (split)
    (2,  0.00,  0.00,  0.00, 0.00, -0.20,  0.50, 0.20),  # inverted V
    (1,  0.00,  0.00,  0.00, 0.00,  0.20, -0.20, 0.10),  # tiny tuck
    # head tilts — curious
    (2,  0.00,  0.00,  0.14, 0.00,  0.00,  0.00, 0.40),  # tilt L
    (2,  0.00,  0.00, -0.14, 0.00,  0.00,  0.00, 0.40),  # tilt R
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

# State-dependent gesture rate (events per second)
GESTURE_RATE = {
    "idle":      0.75,
    "listening": 1.75,
    "thinking":  1.15,
    "speaking":  2.00,
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
    "idle":      {"target_lo": -0.30, "target_hi":  0.35, "interval_lo": 0.20, "interval_hi": 0.50},
    "listening": {"target_lo":  0.10, "target_hi":  0.60, "interval_lo": 0.10, "interval_hi": 0.25},
    "thinking":  {"target_lo": -0.15, "target_hi":  0.45, "interval_lo": 0.15, "interval_hi": 0.35},
    "speaking":  {"target_lo": -0.05, "target_hi":  0.55, "interval_lo": 0.08, "interval_hi": 0.22},
}
ANTENNA_TAU = 0.05         # seconds — smoothing toward the new target
ANTENNA_NEUTRAL_TAU = 1.2  # seconds — slow pull back to state neutral (avoids drift)


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

    def update(self, t: float, state: str, dt: float) -> tuple[float, float, float, float, float, float]:
        """Schedule any due gestures; return total offset to add to base values."""
        self._maybe_schedule(t, state)
        po, yo, ro, byo, alo, aro = self._accumulate(t)
        al_off, ar_off = self._antenna_walk(t, state, dt)
        return po, yo, ro, byo, alo + al_off, aro + ar_off

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


# ── Animator class ────────────────────────────────────────────────────────────

class Animator:
    """Runs animation in a background thread. Call set_state() to switch modes."""

    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"

    def __init__(self, mini, moves_library=None, aliveness: bool = True):
        self.mini      = mini
        self.state     = self.IDLE
        self.aliveness = aliveness
        self._moves    = moves_library  # RecordedMoves HF library, optional
        self._gesture_active = False
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self._t        = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def set_state(self, state):
        with self._lock:
            self.state = state

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
            with self._lock:
                state = self.state
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
                    al =  0.20 + _s(0.15, 0.35, t)
                    ar =  0.20 + _s(0.15, 0.35, t, phase=1.2)

                elif state == self.LISTENING:
                    # head tilted, antennas perked with alternating flutter
                    p  =  0.12 + _s(0.05, 0.40, t)
                    y  =  _s(0.26, 0.35, t) + _s(0.09, 0.79, t)
                    r  =  0.10 + _s(0.05, 0.29, t)
                    by =  _s(0.18, 0.18, t) + _s(0.05, 0.43, t)
                    al =  0.65 + _s(0.18, 0.50, t)
                    ar =  0.35 + _s(0.18, 0.50, t, phase=math.pi)

                elif state == self.THINKING:
                    # quick head nods + rapid antenna alternation (computing feel)
                    p  = -0.05 + _s(0.07, 1.40, t) + _s(0.03, 2.30, t)
                    y  =  _s(0.14, 0.90, t) + _s(0.06, 1.70, t)
                    r  =  _s(0.06, 1.20, t) + _s(0.02, 2.10, t)
                    by =  _s(0.10, 0.55, t)
                    al =  0.45 + _s(0.30, 1.80, t)
                    ar =  0.45 + _s(0.30, 1.80, t, phase=math.pi)

                elif state == self.SPEAKING:
                    # expressive talking — big head bobs, antennas flapping enthusiastically
                    p  =  0.08 + _s(0.10, 0.50, t) + _s(0.04, 1.23, t)
                    y  =  _s(0.26, 0.38, t) + _s(0.10, 0.87, t)
                    r  =  _s(0.08, 0.27, t) + _s(0.03, 0.63, t)
                    by =  _s(0.35, 0.22, t) + _s(0.10, 0.51, t)
                    al =  0.40 + _s(0.35, 0.65, t)
                    ar =  0.40 + _s(0.35, 0.65, t, phase=math.pi * 0.6)

                else:
                    p = y = r = by = al = ar = 0.0

                # ── Aliveness layer — random micro-gestures + antenna walk ──
                if aliveness is not None:
                    po, yo, ro, byo, alo, aro = aliveness.update(t, state, dt)
                    p  += po
                    y  += yo
                    r  += ro
                    by += byo
                    al += alo
                    ar += aro

                _send(self.mini, p, y, r, by, al, ar)
                consecutive_errors = 0

            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    self._stop.set()
                    break

            time.sleep(dt)
            t += dt
