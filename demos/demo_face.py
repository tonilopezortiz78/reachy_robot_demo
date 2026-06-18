"""
demo_face.py — Face Tracking
=============================
Reachy Mini watches for a face via its head camera and follows it in real time.
Head yaw/pitch track the face directly; body yaw slowly rotates to absorb large
horizontal offsets so the head stays centred in its travel range.

Reactions:
  Face found   → antennas rise, head tracks smoothly
  Face lost    → antennas droop, head drifts back to centre
  First detect → excited antenna flutter

Run:  ./run.sh demos/demo_face.py
Press Ctrl-C to stop.
"""
import math
import socket
import subprocess
import time

import cv2

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAM_DEV    = "/dev/video2"
CAM_W, CAM_H = 640, 360     # detection resolution — lower = faster loop
FPS_TARGET = 20             # control loop Hz

# Proportional gains — how far the head moves per unit of normalised error
YAW_GAIN    = 0.90          # head yaw:   error ±1 → ±0.9 rad (±52°)
PITCH_GAIN  = 0.38          # head pitch: error ±1 → ±0.38 rad (±22°)
BODY_GAIN   = 0.80          # body yaw added for large horizontal offsets

# Smoothing — lower alpha = smoother but more lag
HEAD_ALPHA  = 0.18          # head position filter
BODY_ALPHA  = 0.06          # body rotates slower than head

# Antenna angles
ANT_EXCITED  =  0.70        # up when face detected
ANT_IDLE     =  0.15        # neutral / searching
ANT_DROOP    = -0.25        # down when face lost for a while

LOST_TIMEOUT = 2.5          # seconds before "face lost" behaviour kicks in

# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

def start_daemon():
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

# ---------------------------------------------------------------------------
# Sound
# ---------------------------------------------------------------------------

SPEAKER = "plughw:2,0"

def _blip(freq, dur=0.07, vol=0.5):
    subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"aevalsrc=sin(2*PI*{freq}*t)*{vol}:c=mono:s=22050",
         "-t", str(dur), "-f", "alsa", SPEAKER],
    )

def found_chirp():
    """Rising chirp — face acquired."""
    subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "aevalsrc=sin(2*PI*(600+800*t)*t)*0.55:c=mono:s=22050",
         "-t", "0.14", "-f", "alsa", SPEAKER],
    )

def lost_chirp():
    """Falling chirp — face lost."""
    subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "aevalsrc=sin(2*PI*(900-700*t)*t)*0.45:c=mono:s=22050",
         "-t", "0.12", "-f", "alsa", SPEAKER],
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Face Tracking Demo — Ctrl-C to stop")

    # OpenCV Haar cascade — bundled with cv2, no download needed
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Haar cascade not found — reinstall opencv-python")

    print("  Starting daemon...")
    daemon_proc = start_daemon()
    print("  Daemon ready.")

    cap = cv2.VideoCapture(CAM_DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAM_DEV}")
    print(f"  Camera {CAM_DEV} open — {int(cap.get(3))}x{int(cap.get(4))}")

    try:
        with ReachyMini(connection_mode="localhost_only",
                        media_backend="no_media",
                        spawn_daemon=False) as mini:
            mini.wake_up()
            print("  Watching for faces...\n")

            target_yaw   = 0.0
            target_pitch = 0.0
            target_body  = 0.0
            ant_target   = ANT_IDLE

            face_seen   = False
            last_seen_t = 0.0
            loop_dt     = 1.0 / FPS_TARGET

            try:
                while True:
                    t0 = time.time()

                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue

                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5,
                        minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE,
                    )

                    now = time.time()

                    if len(faces) > 0:
                        # Pick the largest face
                        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

                        # Centre in normalised coords: 0 = frame centre, ±1 = edge
                        cx    = (x + w / 2.0) / CAM_W
                        cy    = (y + h / 2.0) / CAM_H
                        err_x = (cx - 0.5) * 2.0   # >0 = face right of centre
                        err_y = (cy - 0.5) * 2.0   # >0 = face below centre

                        new_yaw   = err_x * YAW_GAIN
                        new_pitch = err_y * PITCH_GAIN
                        new_body  = err_x * BODY_GAIN

                        target_yaw   = HEAD_ALPHA * new_yaw   + (1 - HEAD_ALPHA) * target_yaw
                        target_pitch = HEAD_ALPHA * new_pitch + (1 - HEAD_ALPHA) * target_pitch
                        target_body  = BODY_ALPHA * new_body  + (1 - BODY_ALPHA) * target_body

                        ant_target  = ANT_EXCITED
                        last_seen_t = now

                        if not face_seen:
                            print(f"  ✓ Face detected!")
                            found_chirp()
                            face_seen = True

                    else:
                        elapsed_lost = now - last_seen_t
                        if face_seen and elapsed_lost > LOST_TIMEOUT:
                            print("  ✗ Face lost — searching...")
                            lost_chirp()
                            face_seen = False

                        if elapsed_lost > LOST_TIMEOUT:
                            target_yaw   *= 0.97
                            target_pitch *= 0.97
                            target_body  *= 0.95
                            ant_target = ANT_DROOP
                        else:
                            ant_target = ANT_EXCITED if face_seen else ANT_IDLE

                    yaw   = max(-1.50, min(1.50, target_yaw))
                    pitch = max(-0.36, min(0.36, target_pitch))
                    body  = max(-1.40, min(1.40, target_body))

                    mini.set_target(
                        head=create_head_pose(pitch=pitch, yaw=yaw, degrees=False),
                        antennas=[ant_target, ant_target],
                        body_yaw=body,
                    )

                    elapsed = time.time() - t0
                    sleep_t = loop_dt - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)

            except KeyboardInterrupt:
                print("\n  Stopping...")

    finally:
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
