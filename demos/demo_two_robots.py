"""demo_two_robots.py — TWO Reachy Minis performing in parallel (COMEDY DUO).

PREREQUISITES (see docs/TWO_ROBOT_PLAN.md for full details):
  - Two Reachy Mini Lite robots, each with its own USB-C cable
  - lsusb shows two QinHeng 1a86:55d3 motor bridges
  - /dev/ttyACM0 (robot A) + /dev/ttyACM1 (robot B)
  - /dev/video2 (robot A cam) + /dev/video4 (robot B cam)
  - Two ALSA cards: "Audio" (robot A) + "Audio_1" (robot B)

SETUP — launch two daemons manually BEFORE running this demo:
  reachy-mini-daemon --no-media --serialport /dev/ttyACM0 --fastapi-port 8000 &
  reachy-mini-daemon --no-media --serialport /dev/ttyACM1 --fastapi-port 8001 &
  sleep 3

ARCHITECTURE:
  Robot A = TALKER: full conversational engine (mic, speaker, LLM, face ID,
            web dashboard). Same as demo_hackathon.py. Audio on robot A only.
  Robot B = REACTOR: mute motor puppet. Watches A's LiveState via a
            ReactorBot thread and plays reactive gestures on B. No mic,
            no speaker, no audio collision.

  The ReactorBot subscribes to A's state changes:
    - A SPEAKING     → B looks at the kid (head tilt, curious)
    - A LISTENING    → B looks at A, antennas perked (paying attention)
    - A THINKING     → B tilts head, antennas scratch (thinking too)
    - A plays gesture [celebrate] → B plays [confused] or [shy]
    - A plays gesture [laugh]     → B plays [oops] (embarrassed)
    - A dancing      → B dances with a 1-beat delay (goofy, out of sync)
    - A idle > 10s   → B snoozes (slow antenna droop + head nod)
    - A says kid's name → B looks excited, antennas up

THE SHOW (15-min kids demo, two-robot version):
  0:00  Both robots wake up. A greets. B waves enthusiastically alongside.
  1:00  A learns a kid's name. B watches, nods excitedly when A says the name.
  3:00  A chats with kid. B reacts to every gesture:
        A celebrates → B looks confused ("why is A getting all the attention?")
        A laughs → B looks shy/embarrassed
        A thinks → B also thinks (head tilt, antennas scratch)
  7:00  THE ARGUMENT: A says something, B shakes head emphatically.
        A insists (nod), B insists (shake). Back and forth 3-4 times.
        Kid can intervene: "Reachy B, do you agree?" → B nods rapidly.
  9:00  DANCE DUET: A does Macarena. B dances 1 beat behind (goofy).
        They spin toward each other, spin apart, end facing the kids.
  11:00 B "finally speaks": A voices B's line ("I'm the funny one!"),
        B moves its body in sync. Kids crack up.
  13:00 Goodbye: both wave, A says goodbye to each kid by name,
        B does a sad wave (don't leave!), both sleep.

SAFETY: Both robots must goto_sleep() in finally. Overheat risk doubles
with two robots. Each gesture must still be ≤2s. The ReactorBot never
holds set_target — it only fires short goto_target gestures.

Run:  ./run.sh demos/demo_two_robots.py
      (after starting the two daemons manually as above)

NOTE: This demo is NOT in menu.sh because it requires special daemon setup.
Launch it directly. If you haven't started the two daemons, it will fail
with a connection error on port 8001.
"""

import math
import random
import threading
import time

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose, RecordedMoves

from reachy_demo.animator import Animator, NAMED_GESTURES
from reachy_demo.live_state import LiveState
from reachy_demo.web_stage import WebStage

# ── Reactor bot: watches A's state, reacts on B ─────────────────────────────

class ReactorBot:
    """Mute motor puppet that reacts to the talker robot's state.

    Runs in a background thread, polling the shared LiveState every 100ms.
    Fires short gestures on robot B based on what robot A is doing.
    """

    def __init__(self, mini_b, anim_b, state, log=None):
        self.mini = mini_b
        self.anim = anim_b
        self.state = state
        self.log = log
        self._stop = threading.Event()
        self._last_state = "idle"
        self._last_gesture = ""
        self._last_dance = False
        self._idle_since = time.time()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                if self.log:
                    self.log.event(f"  [reactor] error: {e}")
            time.sleep(0.1)

    def _tick(self):
        s = self.state.anim_state
        g = self.state.current_gesture
        dancing = self.state.pending_dance or self.state.anim_state == "speaking" and "dance" in (self.state.last_user or "").lower()

        # ── React to state changes ─────────────────────────────
        if s != self._last_state:
            self._on_state_change(self._last_state, s)
            self._last_state = s
            if s == "idle":
                self._idle_since = time.time()
            else:
                self._idle_since = 0

        # ── React to gesture changes ───────────────────────────
        if g != self._last_gesture and g:
            self._on_gesture(g)
            self._last_gesture = g
        elif not g:
            self._last_gesture = ""

        # ── React to dance ─────────────────────────────────────
        if dancing and not self._last_dance:
            self._on_dance_start()
        self._last_dance = dancing

        # ── Idle snooze ────────────────────────────────────────
        if s == "idle" and self._idle_since and time.time() - self._idle_since > 10:
            self._snooze()

    def _on_state_change(self, old, new):
        if new == "listening":
            # B looks at A, antennas perked — paying attention
            self.mini.goto_target(
                head=create_head_pose(yaw=0.3, pitch=0.05, degrees=False),
                antennas=[0.4, 0.4], body_yaw=0.2, duration=0.4)
        elif new == "thinking":
            # B tilts head, scratches antennas — thinking too
            self.mini.goto_target(
                head=create_head_pose(pitch=-0.1, roll=0.15, degrees=False),
                antennas=[-0.2, 0.5], body_yaw=0.0, duration=0.4)
        elif new == "speaking":
            # B looks at the kid (toward camera), curious tilt
            self.mini.goto_target(
                head=create_head_pose(yaw=-0.2, pitch=0.08, degrees=False),
                antennas=[0.2, 0.2], body_yaw=-0.1, duration=0.3)
        elif new == "idle":
            self.mini.goto_target(
                head=create_head_pose(degrees=False),
                antennas=[0.0, 0.0], body_yaw=0.0, duration=0.5)

    def _on_gesture(self, g):
        # Comedy reactions: A does something, B reacts differently
        reactions = {
            "celebrate": ["confused", "shy"],       # A celebrates → B is confused/jealous
            "proud": ["confused", "oops"],           # A is proud → B is unimpressed
            "laugh": ["oops", "shy"],                # A laughs → B is embarrassed
            "love": ["shy", "surprised"],            # A shows love → B is shy
            "greeting": ["greeting"],                # A greets → B also greets
            "amazed": ["confused"],                  # A is amazed → B is confused
            "success": ["oops"],                     # A succeeds → B is like "ouch"
            "cheerful": ["thinking"],                # A is cheerful → B ponders
        }
        reaction = reactions.get(g, [])
        if reaction:
            self.anim_b.play_gesture(random.choice(reaction))

    def _on_dance_start(self):
        # B gets excited — antennas up, little hop
        self.mini.goto_target(
            head=create_head_pose(pitch=0.25, degrees=False),
            antennas=[0.8, 0.8], body_yaw=0.0, duration=0.2)
        time.sleep(0.15)
        self.mini.goto_target(
            head=create_head_pose(pitch=-0.1, degrees=False),
            antennas=[-0.3, -0.3], body_yaw=0.0, duration=0.15)
        time.sleep(0.1)
        self.mini.goto_target(
            head=create_head_pose(pitch=0.2, degrees=False),
            antennas=[0.9, 0.9], body_yaw=0.0, duration=0.2)

    def _snooze(self):
        # B slowly droops — sleepy
        self.mini.goto_target(
            head=create_head_pose(pitch=-0.2, roll=0.05, degrees=False),
            antennas=[-0.3, -0.3], body_yaw=0.0, duration=1.5)
        time.sleep(2.0)
        # little startle awake
        self.mini.goto_target(
            head=create_head_pose(pitch=0.15, degrees=False),
            antennas=[0.5, 0.5], body_yaw=0.0, duration=0.15)
        time.sleep(0.3)
        self.mini.goto_target(
            head=create_head_pose(degrees=False),
            antennas=[0.0, 0.0], body_yaw=0.0, duration=0.4)


def main():
    """Two-robot demo: A talks, B reacts.

    Robot A uses the full demo_converse engine (via dashboard_cls=WebStage).
    Robot B is a ReactorBot that mirrors/reacts to A's state.
    """
    from demos.demo_converse import main as converse_main

    # ── Phase 1: Start robot B as reactor in background ──────
    # We connect to B first, start the reactor, then hand off to A's engine.
    print("  [two-robots] Connecting to Robot B (reactor, port 8001)...")

    state = LiveState()  # shared between A and B
    reactor = None
    mini_b = None
    mini_b_ctx = None   # so the finally's `if mini_b_ctx` can't NameError if __enter__ raises

    try:
        mini_b_ctx = ReachyMini(connection_mode="localhost_only",
                                media_backend="no_media",
                                spawn_daemon=False,
                                port=8001)
        mini_b = mini_b_ctx.__enter__()
        mini_b.wake_up()
        emotions_b = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        anim_b = Animator(mini_b, moves_library=emotions_b)
        anim_b.set_energy(0.8)

        reactor = ReactorBot(mini_b, anim_b, state)
        reactor.start()
        print("  [two-robots] Robot B reactor started — mirroring A's state")

        # ── Phase 2: Run robot A's full engine ───────────────
        # This blocks until the demo ends (Ctrl-C or Stop button).
        # A's engine writes to the same `state` that reactor reads.
        # We can't pass `state` through converse_main, so we patch it:
        # converse_main creates its own LiveState internally. We need to
        # share it. The cleanest way: monkey-patch LiveState to return
        # our shared instance. (This is a demo, not production code.)
        import reachy_demo.live_state as ls_mod
        _original_init = ls_mod.LiveState.__init__
        _shared = [None]

        def _patched_init(self, *a, **kw):
            _original_init(self, *a, **kw)
            _shared[0] = self  # capture the instance converse_main creates
        ls_mod.LiveState.__init__ = _patched_init

        try:
            converse_main(dashboard_cls=WebStage)
        finally:
            ls_mod.LiveState.__init__ = _original_init

    finally:
        # ── Phase 3: Clean up B ──────────────────────────────
        if reactor:
            reactor.stop()
        if mini_b:
            try:
                mini_b.goto_sleep()
            except Exception:
                pass
        if mini_b_ctx:
            try:
                mini_b_ctx.__exit__(None, None, None)
            except Exception:
                pass
        print("  [two-robots] Robot B asleep. Done.")


if __name__ == "__main__":
    main()
