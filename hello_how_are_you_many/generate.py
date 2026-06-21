"""
Generate "hello, how are you" in many languages with edge-tts.

Uses the same voice + cute-child pitch tuning as reachy_demo.tts_edge, so the
output sounds like Reachy out of the box. Output is 48kHz mono WAV (what
aplay -D plughw:Audio,DEV=0 wants).

Usage:
  ./run.sh generate.py                  # generate all languages
  ./run.sh generate.py --list           # show the language list, no synth
  ./run.sh generate.py spanish japanese # only these (substring match)
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Voice settings (mirror reachy_demo/tts_edge.py) ───────────────────────────
VOICE = "en-US-AvaMultilingualNeural"
RATE = "+30%"
PITCH = "+32Hz"
VOL = "2.5"
SAMPLE_RATE = 48000

# ── Output location ───────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "wavs"
OUT_DIR.mkdir(exist_ok=True)

# ── "Hello, how are you" in many languages ────────────────────────────────────
# Filename is the English name of the language (lowercased, ASCII). The phrase
# is written as a native speaker would actually say it — accents/diacritics
# preserved as UTF-8.
PHRASES: dict[str, str] = {
    "english":    "Hello, how are you?",
    "spanish":    "Hola, ¿cómo estás?",
    "french":     "Bonjour, comment ça va ?",
    "german":     "Hallo, wie geht es dir?",
    "italian":    "Ciao, come stai?",
    "portuguese": "Olá, como você está?",
    "dutch":      "Hallo, hoe gaat het?",
    "swedish":    "Hej, hur mär du?",
    "norwegian":  "Hei, hvordan har du det?",
    "danish":     "Hej, hvordan har du det?",
    "finnish":    "Hei, mitä kuuluu?",
    "polish":     "Cześć, jak się masz?",
    "czech":      "Ahoj, jak se máš?",
    "slovak":     "Ahoj, ako sa máš?",
    "hungarian":  "Szia, hogy vagy?",
    "romanian":   "Salut, ce mai faci?",
    "bulgarian":  "Здравей, как си?",
    "croatian":   "Bok, kako si?",
    "greek":      "Γεια σου, πώς είσαι;",
    "russian":    "Привет, как дела?",
    "ukrainian":  "Привіт, як справи?",
    "turkish":    "Merhaba, nasılsın?",
    "arabic":     "مرحبا، كيف حالك؟",
    "hebrew":     "שלום, מה שלומך?",
    "persian":    "سلام، حال شما چطور است؟",
    "hindi":      "नमस्ते, आप कैसे हैं?",
    "bengali":    "হ্যালো, আপনি কেমন আছেন?",
    "tamil":      "வணக்கம், எப்படி இருக்கிறீர்கள்?",
    "chinese":    "你好，你好吗？",
    "japanese":   "こんにちは、お元気ですか？",
    "korean":     "안녕하세요, 잘 지내세요?",
    "thai":       "สวัสดี คุณสบายดีไหม",
    "vietnamese": "Xin chào, bạn khỏe không?",
    "indonesian": "Halo, apa kabar?",
    "malay":      "Hai, apa khabar?",
    "filipino":   "Kumusta, kumusta ka?",
    "swahili":    "Habari, habari yako?",
    "catalan":    "Hola, com estàs?",
}


# ── Core synth (one language) ────────────────────────────────────────────────
def synth_one(lang: str, text: str) -> tuple[Path, bool]:
    """Synthesise `text` and return (wav_path, was_skipped)."""
    out_wav = OUT_DIR / f"{lang}.wav"
    if out_wav.exists() and out_wav.stat().st_size > 0:
        return out_wav, True

    mp3 = Path(tempfile.mkstemp(suffix=".mp3")[1])
    try:
        # 1. edge-tts → MP3
        subprocess.run(
            ["edge-tts", "--voice", VOICE, "--rate", RATE, "--pitch", PITCH,
             "--text", text, "--write-media", str(mp3)],
            check=True, capture_output=True,
        )
        # 2. ffmpeg → 48kHz mono WAV + volume
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(mp3),
             "-af", f"aresample=resampler=swr:out_sample_rate={SAMPLE_RATE},"
                    f"volume={VOL}",
             "-ac", "1",
             str(out_wav)],
            check=True, capture_output=True,
        )
    finally:
        mp3.unlink(missing_ok=True)
    return out_wav, False


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("filter", nargs="*",
                    help="Only run languages whose name contains any of these "
                         "substrings (case-insensitive). Default: all.")
    ap.add_argument("--list", action="store_true",
                    help="Print the language list and exit.")
    ap.add_argument("--jobs", "-j", type=int, default=4,
                    help="Parallel synth jobs (default 4).")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if output WAV already exists.")
    args = ap.parse_args()

    if args.list:
        for k, v in PHRASES.items():
            print(f"  {k:14s}  {v}")
        return 0

    wanted = PHRASES
    if args.filter:
        flt = [f.lower() for f in args.filter]
        wanted = {k: v for k, v in PHRASES.items()
                  if any(f in k.lower() for f in flt)}
        if not wanted:
            print(f"No languages matched {args.filter!r}", file=sys.stderr)
            return 1

    if not shutil.which("edge-tts"):
        print("edge-tts not on PATH (use ./run.sh …)", file=sys.stderr)
        return 1
    if not shutil.which("ffmpeg"):
        print("ffmpeg not on PATH", file=sys.stderr)
        return 1

    if args.force:
        for lang in wanted:
            (OUT_DIR / f"{lang}.wav").unlink(missing_ok=True)

    print(f"voice={VOICE}  rate={RATE}  pitch={PITCH}  vol={VOL}  "
          f"sr={SAMPLE_RATE}  →  {OUT_DIR}", flush=True)
    print(f"generating {len(wanted)} language(s) with {args.jobs} job(s)…",
          flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(synth_one, lang, txt): lang
                for lang, txt in wanted.items()}
        for fut in as_completed(futs):
            lang = futs[fut]
            try:
                p, skipped = fut.result()
                size = p.stat().st_size
                tag = "skip" if skipped else "ok  "
                print(f"  {tag} {lang:14s}  {size/1024:6.1f} KB", flush=True)
            except Exception as e:
                print(f"  FAIL {lang:14s}  {e}", flush=True)
    print(f"done in {time.time()-t0:.1f}s. {len(wanted)} file(s) in {OUT_DIR}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
