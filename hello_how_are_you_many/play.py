"""
play.py — interactive spacebar-driven player for hello_how_are_you_many WAVs.

Reuses the PHRASES table from generate.py so adding a new language there is
enough — no edits needed here.

Keys (case-insensitive):
  space / n / →     play next
  p / b / ←         play previous
  r / enter         replay current
  /                 jump — type a substring of the language name, Enter
  a                 toggle auto-advance (auto-plays every WAV in order)
  l                 list every language + phrase, press Enter to return
  q / esc           quit

Speaker is selected by --speaker {robot,laptop,hdmi}:
  hdmi    → plughw:CARD=PCH,DEV=3      (HDMI output / external monitor —
            default; louder, no risk of mic feedback)
  laptop  → plughw:CARD=PCH,DEV=0      (laptop built-in speakers)
  robot   → plughw:CARD=Audio,DEV=0    (Reachy Mini's own speaker — watch
            Reachy talk out of its mouth)

Run with ./run.sh play.py (just so the venv + tty behave the same as the demos)
or `python3 play.py` — no Python deps are required.
"""

import argparse
import os
import select
import shutil
import subprocess
import sys
import termios
import tty
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate import PHRASES  # type: ignore  # noqa: E402

WAV_DIR = HERE / "wavs"
SPEAKERS = {
    "robot":  "plughw:CARD=Audio,DEV=0",  # Reachy Mini USB speaker
    "laptop": "plughw:CARD=PCH,DEV=0",    # Laptop built-in speaker
    "hdmi":   "plughw:CARD=PCH,DEV=3",    # HDMI output (monitor / TV)
}
APLAY = shutil.which("aplay") or "aplay"

# Build ordered list of (language, phrase, wav_path) for any file that exists.
ITEMS: list[tuple[str, str, Path]] = []
for lang, phrase in PHRASES.items():
    p = WAV_DIR / f"{lang}.wav"
    if p.exists() and p.stat().st_size > 0:
        ITEMS.append((lang, phrase, p))
MISSING = [k for k in PHRASES if not (WAV_DIR / f"{k}.wav").exists()]
if not ITEMS:
    sys.exit("No WAVs in hello_how_are_you_many/wavs/. Run generate.py first.")
if MISSING:
    print(f"note: {len(MISSING)} missing WAVs will be skipped: "
          f"{', '.join(MISSING)}", file=sys.stderr)

# ── ANSI helpers ─────────────────────────────────────────────────────────────
ESC = "\033"
HIDE, SHOW = f"{ESC}[?25l", f"{ESC}[?25h"
BOLD, DIM, RST = f"{ESC}[1m", f"{ESC}[2m", f"{ESC}[0m"
CYAN, YEL, GRN, RED, MAG = (
    f"{ESC}[36m", f"{ESC}[33m", f"{ESC}[32m", f"{ESC}[31m", f"{ESC}[35m",
)
CLEAR, HOME, CLR_EOL = f"{ESC}[2K", f"{ESC}[H", f"{ESC}[K"


def term_cols() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def fit(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def ffprobe_duration(path: Path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=5)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


# ── Main UI ──────────────────────────────────────────────────────────────────
class Player:
    def __init__(self, speaker: str) -> None:
        self.idx = 0
        self.speaker = speaker
        self.proc: subprocess.Popen | None = None
        self.auto = False
        self.search = ""
        self.status = ""

    # ── rendering
    def header(self) -> str:
        lang, phrase, path = self.current()
        dur = ffprobe_duration(path)
        dur_s = f"{dur:.1f}s" if dur else "?s"
        size_kb = path.stat().st_size / 1024
        auto_tag = f"  {MAG}[auto]{RST}" if self.auto else ""
        spk = f"  {DIM}[{self.speaker}]{RST}"
        return (f"{BOLD}{CYAN}{self.idx + 1}/{len(ITEMS)}{RST}  "
                f"{BOLD}{YEL}{lang}{RST}  "
                f"{DIM}({dur_s}, {size_kb:.0f} KB){auto_tag}{spk}")

    def phrase_line(self) -> str:
        return f"  {fit(self.current()[1], term_cols() - 4)}"

    def controls(self) -> str:
        return (f"  {DIM}space next · p prev · r replay · / jump · "
                f"a auto · l list · q quit{RST}")

    def render(self, status: str = "") -> None:
        self.status = status
        out = HOME
        for ln in [self.header(), "", self.phrase_line(), "",
                   self.controls(), "", f"  {status}"]:
            out += ln + "\n" + CLEAR + "\n"
        sys.stdout.write(out)
        sys.stdout.flush()

    def current(self) -> tuple[str, str, Path]:
        return ITEMS[self.idx]

    # ── playback
    def _stop_proc(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def play_current(self, label: str = "play") -> None:
        self._stop_proc()
        _, _, path = self.current()
        self.proc = subprocess.Popen(
            [APLAY, "-D", self.speaker, "-q", str(path)])
        self.render(f"{GRN}▶ {label}{RST}  {DIM}(space=next, p=prev, "
                    f"r=replay, q=quit){RST}")

    def goto(self, new_idx: int, label: str = "") -> None:
        self.idx = new_idx % len(ITEMS)
        self.play_current(label or f"→ {self.current()[0]}")

    # ── key actions
    def do_next(self) -> None:  self.goto(self.idx + 1, "next")
    def do_prev(self) -> None:  self.goto(self.idx - 1, "prev")
    def do_replay(self) -> None: self.play_current("replay")
    def do_toggle_auto(self) -> None:
        self.auto = not self.auto
        self.render(f"auto-advance: {'ON' if self.auto else 'off'}")
    def do_list(self) -> None:
        cols = term_cols()
        sys.stdout.write(HOME + "\033[J")
        print(f"{BOLD}All {len(ITEMS)} languages:{RST}")
        for i, (lang, phrase, _) in enumerate(ITEMS):
            mark = f"{CYAN}▶{RST}" if i == self.idx else " "
            print(f"  {mark} {lang:14s}  {fit(phrase, cols - 22)}")
        try:
            input(f"\n{DIM}press Enter to return…{RST}")
        except EOFError:
            pass
        self.render()
    def do_jump_prompt(self) -> None:
        self.search = ""
        self.render(f"{CYAN}jump:{RST} {self.search}█")
    def do_jump_char(self, ch: str) -> None:
        if ch in ("\x1b", "\x03"):
            self.search = ""
            self.render()
            return
        if ch in ("\r", "\n"):
            q = self.search.strip().lower()
            self.search = ""
            if not q:
                self.render(); return
            for i, (lang, _, _) in enumerate(ITEMS):
                if q in lang.lower():
                    self.goto(i, f"jump → {lang}")
                    return
            self.render(f"{RED}no match for {q!r}{RST}")
            return
        if ch in ("\x7f", "\b"):
            self.search = self.search[:-1]
        elif ch.isprintable():
            self.search += ch
        self.render(f"{CYAN}jump:{RST} {self.search}█")


# ── input helpers
def read_key(fd: int) -> str:
    """Read one keypress; assemble arrow keys (ESC [ X) into a single string."""
    ch = os.read(fd, 1).decode("utf-8", "replace")
    if ch != "\x1b":
        return ch
    # might be ESC alone, or the start of an arrow/PgUp/etc. sequence
    r, _, _ = select.select([sys.stdin], [], [], 0.05)
    if not r:
        return ch
    ch += os.read(fd, 1).decode("utf-8", "replace")
    r, _, _ = select.select([sys.stdin], [], [], 0.02)
    if r:
        ch += os.read(fd, 1).decode("utf-8", "replace")
    return ch


# ── main loop
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", choices=SPEAKERS.keys(), default="hdmi",
                    help="Where to play: 'hdmi' (HDMI output, default), "
                         "'laptop' (built-in speakers), or "
                         "'robot' (Reachy Mini's own speaker).")
    args = ap.parse_args()
    speaker = SPEAKERS[args.speaker]

    if not sys.stdin.isatty():
        sys.exit("play.py needs an interactive tty.")
    if not shutil.which("aplay"):
        sys.exit("aplay not on PATH")

    p = Player(speaker=speaker)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    sys.stdout.write(HIDE)
    p.render(f"{DIM}press {BOLD}space{RST}{DIM} to play current, "
             f"{BOLD}q{RST}{DIM} to quit{RST}")
    try:
        tty.setcbreak(fd)
        last_was_playing = False
        while True:
            # poll for input, but also react to playback finishing
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not r:
                if p.proc and p.proc.poll() is not None:
                    p.proc = None
                    if p.auto:
                        p.do_next()
                    elif last_was_playing:
                        p.render(f"{DIM}finished — space for next, "
                                 f"q to quit{RST}")
                last_was_playing = p.proc is not None
                continue
            ch = read_key(fd)
            # ── jump mode swallows everything
            if p.search or ch == "/" and not p.auto:
                # '/' starts jump; anything else edits it (ch may be '/')
                if p.search == "" and ch != "/":
                    # we were not in jump mode, so '/' is a fresh trigger
                    if ch == "/":
                        p.do_jump_prompt()
                        continue
                if p.search != "" or ch == "/":
                    if ch == "/":
                        p.do_jump_prompt()
                    else:
                        p.do_jump_char(ch)
                    continue
            # ── normal mode
            k = ch.lower()
            if k in ("q", "\x1b"):
                break
            if ch == "\x03":  # Ctrl-C
                break
            if k in (" ", "n", "\r", "\n") or ch == "\x1b[C":  # space/Enter/→/n
                p.do_next()
            elif k in ("p", "b") or ch == "\x1b[D":          # ←
                p.do_prev()
            elif k == "r":
                p.do_replay()
            elif k == "a":
                p.do_toggle_auto()
            elif k == "l":
                p.do_list()
            elif k == "/":
                p.do_jump_prompt()
    finally:
        p._stop_proc()
        termios.tcsetattr(fd, termios.TCSANOW, old)
        sys.stdout.write(SHOW + HOME + "\033[J")
    print(f"bye — left off at {p.idx + 1}/{len(ITEMS)} "
          f"({ITEMS[p.idx][0]}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
