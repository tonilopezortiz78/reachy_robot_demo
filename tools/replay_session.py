#!/usr/bin/env python3
"""
replay_session.py — replay a recorded session on the robot speaker.

Plays back, in turn order, BOTH sides of the conversation:
  turn_NNN.wav      — what the mic heard (the visitor)        [played quietly]
  reply_NNN_S.wav   — what Reachy said back, segment S         [played normally]

So you hear the whole interaction exactly as it happened — perfect for
debugging "it misheard me" or "the reply audio was wrong."

Usage:
    ./run.sh tools/replay_session.py            # replay the most recent session
    ./run.sh tools/replay_session.py 68         # replay session data/68
    ./run.sh tools/replay_session.py --list     # list available sessions

The visitor's mic audio is played through the robot speaker too (slightly
lower volume) so you can compare what Reachy heard vs. what it answered.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SPEAKER = "plughw:CARD=Audio,DEV=0"


def sessions() -> list[int]:
    if not DATA.exists():
        return []
    return sorted(int(p.name) for p in DATA.iterdir()
                  if p.is_dir() and p.name.isdigit())


def play(wav: Path, gain: float = 1.0):
    """Play a wav on the robot speaker, optionally attenuated."""
    if gain == 1.0:
        subprocess.run(["aplay", "-D", SPEAKER, "-q", str(wav)],
                       stderr=subprocess.DEVNULL)
    else:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", str(wav), "-af", f"volume={gain}", "-f", "alsa", SPEAKER],
            stderr=subprocess.DEVNULL)


def main():
    args = [a for a in sys.argv[1:]]
    if "--list" in args:
        s = sessions()
        if not s:
            print("No sessions recorded in data/")
            return
        print("Available sessions:")
        for n in s:
            audio = DATA / str(n) / "audio"
            turns = len(list(audio.glob("turn_*.wav"))) if audio.exists() else 0
            print(f"  {n}  ({turns} turns)  {DATA / str(n)}")
        return

    if args and args[0].isdigit():
        num = int(args[0])
    else:
        s = sessions()
        if not s:
            print("No sessions recorded in data/")
            return
        num = s[-1]

    audio = DATA / str(num) / "audio"
    if not audio.exists():
        print(f"Session {num} has no audio/ dir ({audio})")
        return

    # Collect turn numbers from filenames
    turn_nums = sorted({
        int(m.group(1))
        for f in audio.iterdir()
        if (m := re.match(r"turn_(\d+)\.wav$", f.name))
    })
    if not turn_nums:
        print(f"Session {num} has no turn_*.wav files yet.")
        return

    print(f"Replaying session {num} — {len(turn_nums)} turns. Ctrl-C to stop.\n")
    try:
        for t in turn_nums:
            mic = audio / f"turn_{t:03d}.wav"
            if mic.exists():
                print(f"  turn {t:03d}  ◀ visitor")
                play(mic, gain=0.8)
                time.sleep(0.3)
            replies = sorted(audio.glob(f"reply_{t:03d}_*.wav"))
            for r in replies:
                print(f"           ▶ Reachy: {r.name}")
                play(r)
                time.sleep(0.15)
            time.sleep(0.4)
    except KeyboardInterrupt:
        print("\nStopped.")
    print("\nDone.")


if __name__ == "__main__":
    main()
