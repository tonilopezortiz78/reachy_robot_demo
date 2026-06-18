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
echo "║     Greeting → dance with music → bow out           ║"
echo "║                                                      ║"
echo "║  q) Quit                                             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
read -p "Pick a demo [1/2/q]: " choice

case $choice in
  1) ./run.sh demos/demo_welcome.py ;;
  2) ./run.sh demos/demo_dance.py ;;
  q|Q) echo "bye"; exit 0 ;;
  *) echo "Unknown choice. Run ./run.sh demos/demo_welcome.py or ./run.sh demos/demo_dance.py directly."; exit 1 ;;
esac
