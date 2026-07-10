"""demo_hackathon.py — Network School kids hackathon demo (one robot).

Same proven conversational engine as demo_converse.py (instant talk + face ID
+ kid mode + NS persona + Macarena), but with a DUAL-VIEW web dashboard:

    /#stage   — big, bold, kid-facing page for the PROJECTOR behind the robot.
                Camera feed, "Reachy hears / thinks / says" live captions,
                gesture emoji, state badge. Pure spectacle.

    /#control — operator-facing CONTROL PANEL for your laptop. Wake/sleep/stop,
                "make Reachy say this" puppet box, all 19 gesture buttons,
                dance triggers, kid-mode + mute toggles, latency/cost, roster.

Open http://<laptop-ip>:8080/#stage on the projector (the #stage hash matters —
the bare URL defaults to the control view).
Open http://localhost:8080/#control on your laptop.

LOUD ROOM: launch with REACHY_LOUD_ROOM=1 ./run.sh demos/demo_hackathon.py to
raise the mic/speech-gate floors so Reachy answers the close, deliberate speaker
instead of the whole chattering crowd. Do a 30-second sound check first — if it
stops hearing soft-voiced kids, drop the flag.

Run:  ./run.sh demos/demo_hackathon.py

This entry point SUPERVISES the demo: if the engine ever crashes with an
unexpected error, it relaunches automatically (a clean Ctrl-C or the web Stop
button exits for good). So an unattended 2-hour event survives a one-off fault
without someone running back to the laptop.
"""

import time

from demos.demo_converse import main
from reachy_demo.web_stage import WebStage

# Restart backstop: allow this many crash-restarts inside RESET_WINDOW_S before
# giving up (a genuinely broken setup shouldn't spin forever). A run that stays
# up longer than the window is treated as healthy and resets the counter.
MAX_FAST_RESTARTS = 6
RESET_WINDOW_S = 90.0


def run_supervised() -> None:
    fast_restarts = 0
    while True:
        started = time.time()
        try:
            main(dashboard_cls=WebStage)
            return  # clean exit: Ctrl-C or the web Stop button — we're done
        except KeyboardInterrupt:
            return
        except Exception as e:
            uptime = time.time() - started
            if uptime > RESET_WINDOW_S:
                fast_restarts = 0  # it ran fine for a while; this is a one-off
            fast_restarts += 1
            if fast_restarts > MAX_FAST_RESTARTS:
                print(f"\n*** Demo crashed {fast_restarts} times quickly; giving up. "
                      f"Last error: {e!r}\n", flush=True)
                raise
            print(f"\n*** Demo crashed ({e!r}) after {uptime:.0f}s — "
                  f"restarting (attempt {fast_restarts}/{MAX_FAST_RESTARTS})...\n",
                  flush=True)
            time.sleep(3.0)  # let the daemon/ports/mic settle before relaunch


if __name__ == "__main__":
    run_supervised()
