#!/usr/bin/env bash
# hear.sh — replay exactly what Reachy heard, with the transcript next to each clip.
#
# Usage:
#   tools/hear.sh            # newest session
#   tools/hear.sh 16         # logs/16
#   tools/hear.sh 16 3       # just turn_003 of logs/16
#
# Plays through the robot speaker (plughw:CARD=Audio). Add HEADPHONES=1 to use
# the system default output instead (handy when the robot is unplugged):
#   HEADPHONES=1 tools/hear.sh 16
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${1:-}" ]]; then SESS="logs/$1"; else SESS="$(ls -dt logs/*/ | head -1)"; fi
SESS="${SESS%/}"
[[ -d "$SESS/audio" ]] || { echo "No audio in $SESS"; exit 1; }

if [[ "${HEADPHONES:-0}" == "1" ]]; then DEV=(); else DEV=(-D plughw:CARD=Audio,DEV=0); fi

# Pull "turn_NNN.wav -> transcript" from the JSONL so each clip is labelled.
declare -A TXT
if [[ -f "$SESS/transcript.jsonl" ]]; then
  while IFS=$'\t' read -r f t; do TXT["$f"]="$t"; done < <(
    ./.venv/bin/python - "$SESS/transcript.jsonl" <<'PY'
import json, os, sys
for line in open(sys.argv[1]):
    o = json.loads(line)
    a = o.get("audio")
    if a and o.get("transcript") is not None:
        tag = "[REJECTED] " if o.get("kind") == "rejected_hallucination" else ""
        print(f"{os.path.basename(a)}\t{tag}{o['transcript']}")
PY
  )
fi

echo "== $SESS =="
for w in "$SESS"/audio/turn_*.wav; do
  b="$(basename "$w")"
  n="$(echo "$b" | grep -o '[0-9]\+')"
  if [[ -n "${2:-}" && "$((10#$n))" -ne "$2" ]]; then continue; fi
  printf '\n%s  %s\n' "$b" "${TXT[$b]:-(no transcript)}"
  aplay "${DEV[@]}" -q "$w"
done
