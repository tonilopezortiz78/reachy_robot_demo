"""
reachy_demo/animator.py — Background animation thread for Reachy Mini.

Usage:
    anim = Animator(mini)
    anim.set_state(Animator.LISTENING)
    ...
    anim.stop()
"""

import math
import threading
import time

from reachy_mini.utils import create_head_pose

# ── Internal helpers ──────────────────────────────────────────────────────────

def _s(amp, freq, t, phase=0.0):
    return amp * math.sin(2 * math.pi * freq * t + phase)


def _send(mini, p, y, r, by, ant_l, ant_r):
    mini.set_target(
        head=create_head_pose(pitch=p, yaw=y, roll=r, degrees=False),
        antennas=[ant_l, ant_r], body_yaw=by,
    )

# ── Animator class ────────────────────────────────────────────────────────────

class Animator:
    """Runs animation in a background thread. Call set_state() to switch modes."""

    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"

    def __init__(self, mini):
        self.mini  = mini
        self.state = self.IDLE
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t    = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def set_state(self, state):
        with self._lock:
            self.state = state

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2)

    def _loop(self):
        t = 0.0
        dt = 0.05
        consecutive_errors = 0
        while not self._stop.is_set():
            with self._lock:
                state = self.state
            try:
                if state == self.IDLE:
                    # gentle ambient sway — slightly asymmetric antennas
                    p  =  0.05 + _s(0.06, 0.28, t) + _s(0.02, 0.67, t)
                    y  =  _s(0.20, 0.22, t) + _s(0.07, 0.53, t)
                    r  =  _s(0.06, 0.17, t) + _s(0.02, 0.41, t)
                    by =  _s(0.15, 0.13, t) + _s(0.04, 0.31, t)
                    al =  0.20 + _s(0.15, 0.35, t)
                    ar =  0.20 + _s(0.15, 0.35, t, phase=1.2)
                    _send(self.mini, p, y, r, by, al, ar)

                elif state == self.LISTENING:
                    # head tilted, antennas perked with alternating flutter
                    p  =  0.12 + _s(0.05, 0.40, t)
                    y  =  _s(0.26, 0.35, t) + _s(0.09, 0.79, t)
                    r  =  0.10 + _s(0.05, 0.29, t)
                    by =  _s(0.18, 0.18, t) + _s(0.05, 0.43, t)
                    al =  0.65 + _s(0.18, 0.50, t)
                    ar =  0.35 + _s(0.18, 0.50, t, phase=math.pi)
                    _send(self.mini, p, y, r, by, al, ar)

                elif state == self.THINKING:
                    # quick head nods + rapid antenna alternation (computing feel)
                    p  = -0.05 + _s(0.07, 1.40, t) + _s(0.03, 2.30, t)
                    y  =  _s(0.14, 0.90, t) + _s(0.06, 1.70, t)
                    r  =  _s(0.06, 1.20, t) + _s(0.02, 2.10, t)
                    by =  _s(0.10, 0.55, t)
                    al =  0.45 + _s(0.30, 1.80, t)
                    ar =  0.45 + _s(0.30, 1.80, t, phase=math.pi)
                    _send(self.mini, p, y, r, by, al, ar)

                elif state == self.SPEAKING:
                    # expressive talking — big head bobs, antennas flapping enthusiastically
                    p  =  0.08 + _s(0.10, 0.50, t) + _s(0.04, 1.23, t)
                    y  =  _s(0.26, 0.38, t) + _s(0.10, 0.87, t)
                    r  =  _s(0.08, 0.27, t) + _s(0.03, 0.63, t)
                    by =  _s(0.35, 0.22, t) + _s(0.10, 0.51, t)
                    al =  0.40 + _s(0.35, 0.65, t)
                    ar =  0.40 + _s(0.35, 0.65, t, phase=math.pi * 0.6)
                    _send(self.mini, p, y, r, by, al, ar)

                consecutive_errors = 0

            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    self._stop.set()
                    break

            time.sleep(dt)
            t += dt
