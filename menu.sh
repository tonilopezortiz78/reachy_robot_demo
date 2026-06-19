#!/bin/bash
# Network School — Reachy Mini demo picker
cd "$(dirname "$0")"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          Reachy Mini — Network School Demos          ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  1) Welcome Demo      demo_welcome.py               ║"
echo "║     Greeting + speech + natural animation           ║"
echo "║                                                      ║"
echo "║  2) Full Dance Show   demo_dance.py                 ║"
echo "║     Greeting → Macarena → climax → bow out          ║"
echo "║                                                      ║"
echo "║  3) Face Tracking     demo_face.py                  ║"
echo "║     Robot follows your face with its camera         ║"
echo "║                                                      ║"
echo "║  4) Lost Brother      demo_lost_friend.py           ║"
echo "║     Emotional pitch for NS Robotics Club            ║"
echo "║                                                      ║"
echo "║  5) Chat with Reachy  demo_chat.py                  ║"
echo "║     Talk to the robot — Whisper STT + LLaMA LLM     ║"
echo "║                                                      ║"
echo "║  q) Quit                                             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
read -p "Pick a demo [1-5/q]: " choice

case $choice in
  1) ./run.sh demos/demo_welcome.py ;;
  2) ./run.sh demos/demo_dance.py ;;
  3) ./run.sh demos/demo_face.py ;;
  4) ./run.sh demos/demo_lost_friend.py ;;
  5) ./run.sh demos/demo_chat.py ;;
  q|Q) echo "bye"; exit 0 ;;
  *) echo "Unknown choice."; exit 1 ;;
esac
