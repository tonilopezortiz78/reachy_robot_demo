#!/bin/bash
# Reachy Mini demo runner — sets up PATH for spawn_daemon=True
set -e
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD:${PYTHONPATH}"
exec .venv/bin/python -u "$@"
