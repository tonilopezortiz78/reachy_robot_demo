"""
reachy_demo/tts_piper.py — Piper TTS utilities.

Provides:
  - load_voice(path) → PiperVoice
  - synth_to_file(voice, text) → str  (temp WAV path; caller must delete)
  - synth_and_play(voice, text) → float  (duration in seconds; one-shot play)
"""

import subprocess
import tempfile
import time
import wave
from pathlib import Path

from piper import PiperVoice

from reachy_demo.audio import SPEAKER


def load_voice(path: str) -> PiperVoice:
    """Load a Piper voice model from disk."""
    return PiperVoice.load(path)


def synth_to_file(voice: PiperVoice, text: str) -> str:
    """
    Synthesise text with Piper, apply FX chain, return temp WAV path.
    Caller is responsible for deleting the returned file.

    FX chain: asetrate *1.10, atempo=1.20, volume=2.5,
              vibrato=f=4.0:d=0.04, aecho=0.88:0.90:16:0.30
    """
    sr  = voice.config.sample_rate
    raw = tempfile.mktemp(suffix=".raw.wav")
    out = tempfile.mktemp(suffix=".wav")
    with wave.open(raw, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        for chunk in voice.synthesize(text):
            wf.writeframes(chunk.audio_int16_bytes)
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", raw,
             "-af", (
                 f"asetrate={sr}*1.10,"
                 "atempo=1.12,"
                 "volume=2.0,"
                 "vibrato=f=4.0:d=0.04,"
                 "aecho=0.88:0.90:16:0.30"
             ),
             out],
            check=True,
        )
    finally:
        Path(raw).unlink(missing_ok=True)
    return out


def synth_and_play(voice: PiperVoice, text: str) -> float:
    """
    Synthesise text with Piper, apply FX chain, play on robot speaker.
    Returns duration in seconds. Blocks until playback is done.

    FX chain: asetrate *1.10, atempo=1.12, volume=2.5,
              vibrato=f=4.0:d=0.04, aecho=0.88:0.90:18:0.35
    """
    sr = voice.config.sample_rate
    raw_path = tempfile.mktemp(suffix=".raw.wav")
    out_path  = tempfile.mktemp(suffix=".wav")
    try:
        with wave.open(raw_path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            for chunk in voice.synthesize(text):
                wf.writeframes(chunk.audio_int16_bytes)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", raw_path,
             "-af", (
                 f"asetrate={sr}*1.10,"
                 "atempo=1.04,"
                 "volume=2.0,"
                 "vibrato=f=4.0:d=0.04,"
                 "aecho=0.88:0.90:18:0.35"
             ),
             out_path],
            check=True,
        )
        proc = subprocess.Popen(
            ["aplay", "-D", SPEAKER, "-q", out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        with wave.open(out_path) as wf:
            dur = wf.getnframes() / wf.getframerate()
        proc.wait()
        time.sleep(0.08)
        return dur
    finally:
        Path(raw_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)
