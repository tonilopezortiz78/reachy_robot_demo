#!/usr/bin/env python3
"""
test_speech_gate.py — validate reachy_demo.speech_gate against REAL recorded audio.

Ground truth comes from the session transcripts (logs/*/transcript.jsonl):
  • kind="rejected_hallucination"  → phantom/noise  → gate MUST reject
  • kind="stt" with >=3 words       → clear speech    → gate MUST pass
  • kind="stt" with 1-2 words        → ambiguous (e.g. "Shh", "Thanks") →
                                       reported only, not asserted
Plus synthetic noise (silence / low hum / a click) → gate MUST reject.

Exit 0 only if every hard-labelled clip is classified correctly, so this is a
real regression test for tuning the thresholds in reachy_demo/speech_gate.py.

Run:  ./run.sh tools/test_speech_gate.py
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np
from silero_vad import load_silero_vad

from reachy_demo.speech_gate import is_real_speech

ROOT = Path(__file__).parent.parent
LOGS = ROOT / "logs"


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        return w.readframes(w.getnframes())


def synth(kind: str, rate: int = 16000) -> bytes:
    rng = np.random.default_rng(0)
    if kind == "low_noise":            # ambient hum: quiet white noise
        a = (rng.standard_normal(rate) * 30).astype(np.int16)
    elif kind == "click":              # tap/door: short burst then silence
        a = np.zeros(int(rate * 0.6), dtype=np.int16)
        a[:800] = (rng.standard_normal(800) * 4000).astype(np.int16)
    else:                              # silence
        a = np.zeros(rate, dtype=np.int16)
    return a.tobytes()


def ground_truth() -> dict:
    """Map audio filename -> ('reject'|'pass'|'skip', transcript) from logs."""
    labels = {}
    for f in sorted(LOGS.glob("*/transcript.jsonl")):
        for line in f.read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            audio = d.get("audio")
            if not audio:
                continue
            key = f"{Path(audio).parent.parent.name}/{Path(audio).name}"
            kind = d.get("kind")
            text = (d.get("transcript") or "").strip()
            if kind == "rejected_hallucination":
                labels[key] = ("reject", text)
            elif kind == "stt":
                words = len(text.split())
                labels[key] = ("pass" if words >= 3 else "skip", text)
    return labels


def main():
    vad = load_silero_vad()
    labels = ground_truth()

    print(f"{'clip':<22}{'rms':>7}{'voic':>6}{'peak':>6}{'dur':>5}  "
          f"{'expect':<7}{'got':<7} {'transcript'}")
    print("-" * 92)

    failures = 0
    counts = {"pass": 0, "reject": 0, "skip": 0}

    for p in sorted(LOGS.glob("*/audio/turn_*.wav")):
        key = f"{p.parent.parent.name}/{p.name}"
        expect, text = labels.get(key, ("skip", ""))
        ok, m = is_real_speech(read_wav(p), vad)
        got = "pass" if ok else "reject"
        counts[expect] = counts.get(expect, 0) + 1
        if expect == "skip":
            mark = "·"
        elif expect == got:
            mark = "✓"
        else:
            mark = "✗ MISMATCH"
            failures += 1
        print(f"{key:<22}{m['rms']:>7.0f}{m['voiced_ratio']:>6.2f}"
              f"{m['peak_prob']:>6.2f}{m['duration_s']:>5.1f}  "
              f"{expect:<7}{got:<7} {mark}  {text[:34]!r}")

    print("-" * 92)
    for kind in ("silence", "low_noise", "click"):
        ok, m = is_real_speech(synth(kind), vad)
        if ok:
            failures += 1
            mark = "✗ should reject"
        else:
            mark = "✓"
        print(f"{('noise:'+kind):<22}{m['rms']:>7.0f}{m['voiced_ratio']:>6.2f}"
              f"{m['peak_prob']:>6.2f}{m['duration_s']:>5.1f}  "
              f"{'reject':<7}{'reject':<7} {mark}")

    print("-" * 92)
    print(f"labelled: {counts.get('pass',0)} pass, {counts.get('reject',0)} reject, "
          f"{counts.get('skip',0)} ambiguous (not asserted)")
    if failures:
        print(f"FAILED: {failures} hard-labelled clip(s) misclassified.")
        sys.exit(1)
    print("OK: every hard-labelled clip classified correctly.")


if __name__ == "__main__":
    main()
