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


def _send(mini, p, y, r, by, ant):
    mini.set_target(
        head=create_head_pose(pitch=p, yaw=y, roll=r, degrees=False),
        antennas=[ant, ant], body_yaw=by,
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
                    # gentle ambient sway
                    p  =  0.05 + _s(0.05, 0.28, t) + _s(0.02, 0.67, t)
                    y  =  _s(0.18, 0.22, t) + _s(0.06, 0.53, t)
                    r  =  _s(0.04, 0.17, t)
                    by =  _s(0.12, 0.13, t)
                    a  =  0.10 + _s(0.10, 0.35, t)
                    _send(self.mini, p, y, r, by, a)

                elif state == self.LISTENING:
                    # head tilted, antennas perked, scanning gently
                    p  =  0.10 + _s(0.04, 0.42, t)
                    y  =  _s(0.22, 0.35, t) + _s(0.08, 0.79, t)
                    r  =  0.08 + _s(0.04, 0.31, t)
                    by =  _s(0.14, 0.18, t)
                    a  =  0.60 + _s(0.12, 0.47, t)
                    _send(self.mini, p, y, r, by, a)

                elif state == self.THINKING:
                    # small rapid head nods, antennas wiggle
                    p  = -0.05 + _s(0.06, 1.40, t) + _s(0.02, 2.30, t)
                    y  =  _s(0.12, 0.90, t) + _s(0.05, 1.70, t)
                    r  =  _s(0.05, 1.20, t)
                    by =  _s(0.08, 0.55, t)
                    a  =  0.30 + _s(0.18, 1.50, t)
                    _send(self.mini, p, y, r, by, a)

                elif state == self.SPEAKING:
                    # animated talking motion
                    p  =  0.08 + _s(0.08, 0.50, t) + _s(0.03, 1.23, t)
                    y  =  _s(0.22, 0.38, t) + _s(0.08, 0.87, t)
                    r  =  _s(0.06, 0.27, t) + _s(0.02, 0.63, t)
                    by =  _s(0.28, 0.22, t) + _s(0.08, 0.51, t)
                    a  =  0.35 + _s(0.20, 0.65, t)
                    _send(self.mini, p, y, r, by, a)

                consecutive_errors = 0

            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    self._stop.set()
                    break

            time.sleep(dt)
            t += dt
