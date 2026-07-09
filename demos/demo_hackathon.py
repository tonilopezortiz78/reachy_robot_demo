"""demo_hackathon.py — Network School kids hackathon demo (one robot).

Same proven conversational engine as demo_converse.py (instant talk + face ID
+ kid mode + NS persona + Macarena), but with a DUAL-VIEW web dashboard:

    /stage   — big, bold, kid-facing page for the PROJECTOR behind the robot.
               Camera feed, "Reachy hears / thinks / says" live captions,
               gesture emoji, state badge. Pure spectacle.

    /control — operator-facing CONTROL PANEL for your laptop. Wake/sleep/stop,
               "make Reachy say this" puppet box, all 19 gesture buttons,
               Macarena trigger, kid-mode + mute toggles, latency/cost, roster.

Open http://<laptop-ip>:8080/stage on the projector.
Open http://localhost:8080/control on your laptop.

Run:  ./run.sh demos/demo_hackathon.py
"""

from demos.demo_converse import main
from reachy_demo.web_stage import WebStage

if __name__ == "__main__":
    main(dashboard_cls=WebStage)
