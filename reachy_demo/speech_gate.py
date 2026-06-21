"""
reachy_demo/speech_gate.py — reject background noise BEFORE it reaches Whisper.

The problem: the VAD fires on ambient sound (hum, a door, a distant voice),
the clip goes to Whisper, and Whisper confidently HALLUCINATES words on
near-silence ("Thank you.", "you", "ご視聴ありがとうございました"). The robot
then "hears" things nobody said.

Best practice (per Silero/Whisper community): don't rely on Whisper's own
no_speech_prob — use a standalone VAD pass to decide whether a captured clip is
really speech, and gate on signal energy. This module does exactly that, locally
and in ~10ms, so noise clips are dropped before any API call (faster, cheaper,
and hallucination-proof).

We combine three cheap signals on the captured PCM:
  • rms          — overall energy. Ambient hum on this hardware is RMS ~2–50;
                   a person speaking is hundreds–thousands.
  • voiced_ratio — fraction of 512-sample frames the Silero model scores as
                   speech. Real speech is mostly voiced; noise/clicks are not.
  • peak_prob    — the single most speech-like frame. Requires at least one
                   clearly-voiced moment, so steady hum (no peak) is rejected.

A real utterance must clear ALL of: enough energy, enough voiced frames, one
confident peak, and a minimum duration. Tuned against real recorded sessions
(see tools/test_speech_gate.py).

NOTE: this rejects NOISE, not other humans. Distinguishing the visitor from a
bystander's voice needs speaker recognition (e.g. a 3D-Speaker embedding, like
xiaozhi-esp32 uses) — a possible future add; see is_real_speech docstring.
"""
import numpy as np
import torch

# ── Gate thresholds (tuned on real recorded turns; see test) ───────────────────
MIN_RMS          = 120.0   # below this the clip is ambient hum, not a speaker
MIN_VOICED_RATIO = 0.30    # ≥30% of frames must be voiced
MIN_PEAK_PROB    = 0.75    # ≥1 clearly-voiced frame (rejects steady hum)
MIN_DURATION_S   = 0.30    # shorter than this is a click/blip
_FRAME           = 512     # Silero's native frame size at 16 kHz
_VAD_THRESH      = 0.5     # per-frame "voiced" cutoff for the ratio


def speech_metrics(pcm: bytes, vad_model, rate: int = 16000,
                   frame: int = _FRAME, vad_thresh: float = _VAD_THRESH) -> dict:
    """Compute energy + voiced-frame metrics for a raw PCM clip (int16 mono).
    Returns {rms, voiced_ratio, peak_prob, duration_s, n_frames}. Pure analysis,
    no thresholds applied — see is_real_speech() for the decision."""
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    duration_s = len(arr) / rate
    if arr.size == 0:
        return {"rms": 0.0, "voiced_ratio": 0.0, "peak_prob": 0.0,
                "duration_s": 0.0, "n_frames": 0}

    rms = float(np.sqrt(np.mean(arr ** 2)))
    f32 = arr / 32768.0

    # Sequential per-frame VAD probabilities (Silero is a streaming model, so
    # reset its state first and feed frames in order).
    try:
        vad_model.reset_states()
    except Exception:
        pass
    probs = []
    for i in range(0, len(f32) - frame + 1, frame):
        chunk = torch.from_numpy(f32[i:i + frame])
        with torch.no_grad():
            probs.append(float(vad_model(chunk, rate)))

    if not probs:
        return {"rms": rms, "voiced_ratio": 0.0, "peak_prob": 0.0,
                "duration_s": duration_s, "n_frames": 0}

    p = np.asarray(probs)
    return {
        "rms": rms,
        "voiced_ratio": float((p > vad_thresh).mean()),
        "peak_prob": float(p.max()),
        "duration_s": duration_s,
        "n_frames": len(probs),
    }


def is_real_speech(pcm: bytes, vad_model, rate: int = 16000,
                   min_rms: float = MIN_RMS,
                   min_voiced_ratio: float = MIN_VOICED_RATIO,
                   min_peak_prob: float = MIN_PEAK_PROB,
                   min_duration_s: float = MIN_DURATION_S) -> tuple[bool, dict]:
    """Return (ok, metrics). ok is True only when the clip clears every gate, so
    ambient noise / hum / brief clicks are rejected before any STT call.

    `metrics` includes a "reject_reason" string (empty when ok) for logging.

    Limitation: this rejects NOISE, not a second human voice. If background
    *people* are the problem, add speaker verification (embed the enrolled
    visitor's voice and compare) — out of scope here.
    """
    m = speech_metrics(pcm, vad_model, rate=rate)
    reasons = []
    if m["duration_s"] < min_duration_s:
        reasons.append(f"too short ({m['duration_s']:.2f}s<{min_duration_s})")
    if m["rms"] < min_rms:
        reasons.append(f"too quiet (rms {m['rms']:.0f}<{min_rms:.0f})")
    if m["voiced_ratio"] < min_voiced_ratio:
        reasons.append(f"not voiced enough ({m['voiced_ratio']:.2f}<{min_voiced_ratio})")
    if m["peak_prob"] < min_peak_prob:
        reasons.append(f"no clear voice peak ({m['peak_prob']:.2f}<{min_peak_prob})")
    ok = not reasons
    m["ok"] = ok
    m["reject_reason"] = "; ".join(reasons)
    return ok, m
