#!/bin/bash
# Network School — Reachy Mini demo picker
cd "$(dirname "$0")"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║            Reachy Mini — Network School             ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║                                                      ║"
echo "║  1) Say Hello                                        ║"
echo "║     Reachy greets everyone who walks by             ║"
echo "║                                                      ║"
echo "║  2) Dance Show                                       ║"
echo "║     Full Macarena performance with music             ║"
echo "║                                                      ║"
echo "║  3) Recognise Faces                                  ║"
echo "║     Reachy learns your name and remembers you        ║"
echo "║                                                      ║"
echo "║  4) Talk to Reachy  ⚡ Fast                          ║"
echo "║     Ask anything · any language · ask it to dance   ║"
echo "║     Replies in ~1 s                                  ║"
echo "║                                                      ║"
echo "║  5) Talk to Reachy  🧠 Smart                         ║"
echo "║     Same as above but deeper, smarter answers        ║"
echo "║     Replies in ~15 s                                 ║"
echo "║                                                      ║"
echo "║  6) Talk to Reachy  🚀 Instant                       ║"
echo "║     Streaming voice — starts talking almost          ║"
echo "║     immediately (~0.4 s to first word)               ║"
echo "║                                                      ║"
echo "║  7) Reachy Unified  🤖💬                             ║"
echo "║     Instant talk + face ID + web dashboard          ║"
echo "║     Camera + status at  http://localhost:8080       ║"
echo "║     Faces: drop photos in faces/<name>/             ║"
echo "║                                                      ║"
echo "║  8) Kids Hackathon  🎉👤🎛️                           ║"
echo "║     Unified demo + dual-view tabbed dashboard       ║"
echo "║     /#stage on projector · /#control on laptop      ║"
echo "║     Puppet panel: gestures, dance, say-anything     ║"
echo "║     Kid mode on · NS persona · face ID              ║"
echo "║                                                      ║"
echo "║  q) Quit                                             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
read -p "Pick [1-8/q]: " choice

case $choice in
  1) ./run.sh demos/demo_welcome.py ;;
  2) ./run.sh demos/demo_dance.py ;;
  3) ./run.sh demos/demo_face_recognition.py ;;
  4) ./run.sh demos/demo_tools7.py ;;
  5) ./run.sh demos/demo_deepseek.py ;;
  6) ./run.sh demos/demo_instant.py ;;
  7) ./run.sh demos/demo_converse.py ;;
  8) ./run.sh demos/demo_hackathon.py ;;
  q|Q) echo "Bye!"; exit 0 ;;
  *) echo "Please pick 1-8 or q."; exit 1 ;;
esac
