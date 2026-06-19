"""
demo_face_recognition.py — Face Recognition + Greeting
=======================================================
Loads a roster of known faces from  faces/<name>/*.jpg  (or .png).
When a known face is detected Reachy greets them by name via TTS.
Unknown visitors get a generic welcome. Head tracks the face in real time.

Roster setup:
  mkdir -p faces/tony
  cp your_photo.jpg faces/tony/
  # Add as many photos per person as you like (more = more reliable)

Run:  ./run.sh demos/demo_face_recognition.py
Press q in the preview window or Ctrl-C to stop.
"""

import random
import socket
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import cv2
import numpy as np

try:
    import face_recognition
except ImportError:
    raise SystemExit(
        "face_recognition not installed.\n"
        "Run:  .venv/bin/pip install face_recognition\n"
        "(Needs cmake and dlib — both already present on this machine.)"
    )

from piper import PiperVoice

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent.parent
FACES_DIR  = ROOT / "faces"
VOICE_PATH = str(ROOT / "voices" / "en_US-amy-medium.onnx")
SPEAKER    = "plughw:CARD=Audio,DEV=0"
CAM_DEV    = "/dev/video2"

# ── Camera / recognition config ───────────────────────────────────────────────

CAM_W, CAM_H  = 640, 360
RECOG_SCALE   = 0.5     # downsample frame for recognition speed
TOLERANCE     = 0.52    # face distance threshold — lower = stricter
GREET_COOLDOWN = 90.0   # seconds before re-greeting the same person

# ── Head tracking ─────────────────────────────────────────────────────────────

YAW_GAIN   = 0.90
PITCH_GAIN = 0.38
BODY_GAIN  = 0.80
HEAD_ALPHA = 0.18
BODY_ALPHA = 0.06
LOST_TIMEOUT = 3.0

ANT_EXCITED =  0.70
ANT_IDLE    =  0.15
ANT_DROOP   = -0.25

# ── Greetings ─────────────────────────────────────────────────────────────────

KNOWN_GREETINGS = [
    "Hey {name}! So great to see you at Network School!",
    "Oh! It's {name}! Hello! You're one of my favourite humans!",
    "Welcome back, {name}! Network School is better with you here!",
    "{name}! My circuits are lighting up — hello!",
]

UNKNOWN_GREETINGS = [
    "Hello there! I'm Reachy, the Network School robot! Welcome!",
    "Hi! I'm Reachy! I don't think we've met yet — welcome to Network School!",
    "Welcome to Network School! I'm Reachy, your friendly robot ambassador!",
    "Hello! I'm Reachy! Ask me anything about Network School, Bitcoin, or AI!",
]

# ── Roster loader ─────────────────────────────────────────────────────────────

def load_roster() -> tuple[list, list]:
    """
    Walk faces/<name>/*.jpg — return (encodings_list, names_list).
    Each photo may yield 0 or 1 encoding; photos with no detectable face are skipped.
    """
    encodings, names = [], []
    if not FACES_DIR.exists():
        print("  [roster] faces/ directory not found — no known faces loaded")
        return encodings, names

    for person_dir in sorted(FACES_DIR.iterdir()):
        if not person_dir.is_dir() or person_dir.name.startswith("."):
            continue
        name = person_dir.name.replace("_", " ").title()
        count = 0
        for img_path in sorted(person_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(img)
            if not encs:
                print(f"  [roster] {img_path.name}: no face detected — skipping")
                continue
            encodings.append(encs[0])
            names.append(name)
            count += 1
        if count:
            print(f"  [roster] {name}: {count} photo(s) loaded")
        else:
            print(f"  [roster] {name}: no usable photos — skipping")

    print(f"  [roster] {len(set(names))} people, {len(encodings)} reference encodings")
    return encodings, names

# ── Daemon ────────────────────────────────────────────────────────────────────

def start_daemon():
    subprocess.run(["pkill", "-9", "-f", "reachy-mini-daemon"], check=False)
    time.sleep(0.3)
    proc = subprocess.Popen(
        ["reachy-mini-daemon", "--no-media"], start_new_session=True,
    )
    for _ in range(30):
        time.sleep(0.5)
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=0.3):
                return proc
        except OSError:
            pass
    raise RuntimeError("Daemon did not start within 15 s")

# ── TTS ───────────────────────────────────────────────────────────────────────

def _synth_and_play(voice: PiperVoice, text: str):
    """Synthesise text with Piper + FX chain and play on robot speaker."""
    sr  = voice.config.sample_rate
    raw = tempfile.mktemp(suffix=".raw.wav")
    out = tempfile.mktemp(suffix=".wav")
    try:
        with wave.open(raw, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            for chunk in voice.synthesize(text):
                wf.writeframes(chunk.audio_int16_bytes)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", raw,
             "-af", (
                 f"asetrate={sr}*1.10,"
                 "atempo=1.08,"
                 "volume=2.0,"
                 "vibrato=f=4.0:d=0.04,"
                 "aecho=0.88:0.90:16:0.30"
             ),
             out],
            check=True,
        )
        finally_path = Path(raw)  # raw cleaned in finally below
        proc = subprocess.Popen(
            ["aplay", "-D", SPEAKER, "-q", out],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.wait()
    finally:
        Path(raw).unlink(missing_ok=True)
        Path(out).unlink(missing_ok=True)

def greet_async(voice: PiperVoice, text: str):
    """Fire-and-forget TTS in a background thread so tracking loop is not blocked."""
    t = threading.Thread(target=_synth_and_play, args=(voice, text), daemon=True)
    t.start()

# ── Sound FX ──────────────────────────────────────────────────────────────────

def found_chirp():
    subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "aevalsrc=sin(2*PI*(600+800*t)*t)*0.55:c=mono:s=22050",
        "-t", "0.14", "-f", "alsa", SPEAKER])

def lost_chirp():
    subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "aevalsrc=sin(2*PI*(900-700*t)*t)*0.45:c=mono:s=22050",
        "-t", "0.12", "-f", "alsa", SPEAKER])

# ── Recognition ───────────────────────────────────────────────────────────────

def identify(frame_rgb, known_encodings, known_names) -> list[tuple[tuple, str, float]]:
    """
    Run face recognition on a (possibly downscaled) RGB frame.
    Returns list of (box_full_res, name, confidence) for each detected face.
    box_full_res is (top, right, bottom, left) scaled back to CAM_W x CAM_H.
    name is the matched name or "unknown".
    confidence is 1 - distance (higher = more confident).
    """
    small = cv2.resize(frame_rgb, (0, 0), fx=RECOG_SCALE, fy=RECOG_SCALE)
    locations = face_recognition.face_locations(small, model="hog")
    if not locations:
        return []

    encodings = face_recognition.face_encodings(small, locations)
    results = []
    inv = 1.0 / RECOG_SCALE

    for (top, right, bottom, left), enc in zip(locations, encodings):
        # Scale box back to full resolution
        box = (int(top * inv), int(right * inv), int(bottom * inv), int(left * inv))

        name = "unknown"
        confidence = 0.0

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, enc)
            best_idx  = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            if best_dist < TOLERANCE:
                name       = known_names[best_idx]
                confidence = 1.0 - best_dist

        results.append((box, name, confidence))

    return results

# ── Overlay drawing ───────────────────────────────────────────────────────────

ARM, GAP = 22, 6

def crosshair(img, px, py, color, thickness=2):
    cv2.line(img, (px - ARM, py), (px - GAP, py), color, thickness)
    cv2.line(img, (px + GAP, py), (px + ARM, py), color, thickness)
    cv2.line(img, (px, py - ARM), (px, py - GAP), color, thickness)
    cv2.line(img, (px, py + GAP), (px, py + ARM), color, thickness)
    cv2.circle(img, (px, py), GAP, color, thickness)

def draw_overlay(frame, face_results, fps, robot_ready, face_seen):
    fh, fw = frame.shape[:2]
    crosshair(frame, fw // 2, fh // 2, (210, 210, 210), 1)

    for (top, right, bottom, left), name, confidence in face_results:
        # Mirror x for display (camera is mirrored)
        ml = fw - right
        mr = fw - left
        color = (0, 220, 0) if name != "unknown" else (0, 180, 255)
        cv2.rectangle(frame, (ml, top), (mr, bottom), color, 2)
        fcx = (ml + mr) // 2
        fcy = (top + bottom) // 2
        crosshair(frame, fcx, fcy, color, 2)

        label = f"{name} ({confidence:.0%})" if name != "unknown" else "visitor"
        cv2.putText(frame, label, (ml, top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if not robot_ready:
        status, color = "STARTING ROBOT...", (0, 200, 255)
    elif face_seen:
        status, color = "FACE LOCKED", (0, 220, 0)
    else:
        status, color = "SEARCHING...", (0, 120, 220)

    cv2.putText(frame, status, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"{fps:.0f} fps", (8, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

    if not FACES_DIR.exists() or not any(
        d.is_dir() and not d.name.startswith(".")
        for d in FACES_DIR.iterdir()
    ):
        cv2.putText(frame, "No roster — add photos to faces/<name>/",
                    (8, fh - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Face Recognition Demo")
    print("  Loading voice...")
    voice = PiperVoice.load(VOICE_PATH)

    print("  Loading face roster...")
    known_encodings, known_names = load_roster()
    if not known_encodings:
        print("  [!] No known faces — will greet everyone as a visitor.")
        print("      Add photos:  mkdir -p faces/yourname && cp photo.jpg faces/yourname/")

    print("  Opening camera...")
    cap = cv2.VideoCapture(CAM_DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {CAM_DEV}")

    WIN = "Reachy — Face Recognition  (q to quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 450)

    print("  Starting daemon...")
    daemon_proc = start_daemon()

    # Tracking state
    target_yaw   = target_pitch = target_body = 0.0
    ant_target   = ANT_IDLE
    face_seen    = False
    last_seen_t  = 0.0
    fps_t        = time.time()
    fps_count    = fps_display = 0.0

    # Recognition state
    last_results: list = []           # (box, name, confidence) per face
    greeted_at:   dict = {}           # name -> timestamp of last greeting
    greeted_unknown_at: float = 0.0   # timestamp of last "unknown visitor" greeting

    mini = None

    try:
        with ReachyMini(connection_mode="localhost_only",
                        media_backend="no_media",
                        spawn_daemon=False) as mini:
            mini.wake_up()
            print("  Robot ready — watching for faces!\n")

            try:
                while True:
                    t0  = time.time()
                    now = t0

                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.02)
                        continue

                    # ── Recognition ─────────────────────────────────────────
                    frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_results = identify(frame_rgb, known_encodings, known_names)

                    # ── Tracking: use largest face ───────────────────────────
                    if last_results:
                        # Largest face by area
                        (top, right, bottom, left), _, _ = max(
                            last_results,
                            key=lambda r: (r[0][2] - r[0][0]) * (r[0][1] - r[0][3])
                        )
                        cx = ((left + right) / 2.0) / CAM_W
                        cy = ((top + bottom) / 2.0) / CAM_H
                        err_x = (cx - 0.5) * 2.0
                        err_y = (cy - 0.5) * 2.0

                        target_yaw   = HEAD_ALPHA * (err_x * YAW_GAIN)  + (1 - HEAD_ALPHA) * target_yaw
                        target_pitch = HEAD_ALPHA * (err_y * PITCH_GAIN) + (1 - HEAD_ALPHA) * target_pitch
                        target_body  = BODY_ALPHA * (err_x * BODY_GAIN)  + (1 - BODY_ALPHA) * target_body
                        ant_target   = ANT_EXCITED
                        last_seen_t  = now

                        if not face_seen:
                            print("  Face detected!")
                            found_chirp()
                            face_seen = True

                        # ── Greet if needed ──────────────────────────────────
                        for _, name, confidence in last_results:
                            if name != "unknown":
                                last_t = greeted_at.get(name, 0.0)
                                if now - last_t > GREET_COOLDOWN:
                                    text = random.choice(KNOWN_GREETINGS).format(name=name)
                                    print(f"  Greeting: {text}")
                                    greet_async(voice, text)
                                    greeted_at[name] = now
                            else:
                                if now - greeted_unknown_at > GREET_COOLDOWN:
                                    text = random.choice(UNKNOWN_GREETINGS)
                                    print(f"  Greeting: {text}")
                                    greet_async(voice, text)
                                    greeted_unknown_at = now

                    else:
                        elapsed_lost = now - last_seen_t
                        if face_seen and elapsed_lost > LOST_TIMEOUT:
                            print("  Face lost — searching...")
                            lost_chirp()
                            face_seen = False

                        if elapsed_lost > LOST_TIMEOUT:
                            target_yaw   *= 0.97
                            target_pitch *= 0.97
                            target_body  *= 0.95
                            ant_target = ANT_DROOP

                    # ── FPS counter ──────────────────────────────────────────
                    fps_count += 1
                    if now - fps_t >= 1.0:
                        fps_display = fps_count / (now - fps_t)
                        fps_count = 0; fps_t = now

                    # ── Display ──────────────────────────────────────────────
                    display = cv2.flip(frame, 1)
                    draw_overlay(display, last_results, fps_display, True, face_seen)
                    cv2.imshow(WIN, display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    # ── Robot ────────────────────────────────────────────────
                    yaw   = max(-1.50, min(1.50, target_yaw))
                    pitch = max(-0.36, min(0.36, target_pitch))
                    body  = max(-1.40, min(1.40, target_body))
                    mini.set_target(
                        head=create_head_pose(pitch=pitch, yaw=yaw, degrees=False),
                        antennas=[ant_target, ant_target],
                        body_yaw=body,
                    )

                    elapsed = time.time() - t0
                    sleep_t = (1.0 / 15) - elapsed   # ~15 fps (recognition is heavy)
                    if sleep_t > 0:
                        time.sleep(sleep_t)

            except KeyboardInterrupt:
                print("\n  Stopping...")
            finally:
                mini.goto_sleep()

    finally:
        cv2.destroyAllWindows()
        cap.release()
        daemon_proc.terminate()
        try:
            daemon_proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            daemon_proc.kill()
            daemon_proc.wait()
        print("  Done.")


if __name__ == "__main__":
    main()
