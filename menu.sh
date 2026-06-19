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
echo "║  3) NS Ambassador     demo_talk_ns.py               ║"
echo "║     Reachy NS ambassador — Piper voice (offline)    ║"
echo "║                                                      ║"
echo "║  4) Face Recognition  demo_face_recognition.py      ║"
echo "║     Greets visitors by name — add photos to faces/  ║"
echo "║                                                      ║"
echo "║  5) NS Ambassador v2  demo_edge.py                  ║"
echo "║     Same as 3 but edge-tts — cuter voice (online)   ║"
echo "║                                                      ║"
echo "║  6) Fluid Dialog     demo_dialog.py                 ║"
echo "║     Fast conversation — barge-in, ~700ms turn take  ║"
echo "║                                                      ║"
echo "║  q) Quit                                             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
read -p "Pick a demo [1-6/q]: " choice

case $choice in
  1) ./run.sh demos/demo_welcome.py ;;
  2) ./run.sh demos/demo_dance.py ;;
  3) ./run.sh demos/demo_talk_ns.py ;;
  4) ./run.sh demos/demo_face_recognition.py ;;
  5) ./run.sh demos/demo_edge.py ;;
  6) ./run.sh demos/demo_dialog.py ;;
  q|Q) echo "bye"; exit 0 ;;
  *) echo "Unknown choice."; exit 1 ;;
esac
