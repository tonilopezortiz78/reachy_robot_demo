#!/bin/bash
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD"
pkill -9 -f "reachy-mini-daemon" 2>/dev/null
sleep 1
.venv/bin/python -u demos/demo_converse.py > /tmp/reachy_converse.log 2>&1 &
echo "PID=$!"
exit 0