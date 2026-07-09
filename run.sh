#!/bin/bash
# Reachy Mini demo runner — sets up PATH/PYTHONPATH for manual-daemon demos
# (spawn_daemon=True is broken; demos use reachy_demo.daemon.start_daemon())
set -e
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD:${PYTHONPATH}"
exec .venv/bin/python -u "$@"
