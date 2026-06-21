#!/bin/bash
# Network School — Reachy Mini demo picker
cd "$(dirname "$0")"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          Reachy Mini — Network School Demos          ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  1) Welcome             demo_welcome.py             ║"
echo "║     Greeting + speech + natural animation           ║"
echo "║                                                      ║"
echo "║  2) Macarena Show       demo_dance.py               ║"
echo "║     Greeting → beat-synced Macarena → climax        ║"
echo "║                                                      ║"
echo "║  3) Face Recognition    demo_face_recognition.py    ║"
echo "║     Greets visitors by name — add photos to faces/  ║"
echo "║                                                      ║"
echo "║  4) NS Ambassador       demo_tools7.py              ║"
echo "║     Multilingual · barge-in · web search · dance    ║"
echo "║     AI picks gesture · AvaMultilingual · Groq LLM   ║"
echo "║                                                      ║"
echo "║  5) NS Ambassador+      demo_deepseek.py            ║"
echo "║     Same as 4 but DeepSeek V4 Flash via opencode    ║"
echo "║     Smarter replies · longer context · web search   ║"
echo "║                                                      ║"
echo "║  q) Quit                                             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
read -p "Pick a demo [1-5/q]: " choice

case $choice in
  1) ./run.sh demos/demo_welcome.py ;;
  2) ./run.sh demos/demo_dance.py ;;
  3) ./run.sh demos/demo_face_recognition.py ;;
  4) ./run.sh demos/demo_tools7.py ;;
  5) ./run.sh demos/demo_deepseek.py ;;
  q|Q) echo "bye"; exit 0 ;;
  *) echo "Unknown choice — pick 1-5 or q."; exit 1 ;;
esac
