"""
demo_face.py — Face Tracking
=============================
Camera feed appears IMMEDIATELY in a window. Daemon starts in the background.
Once the robot is ready, head follows any detected face in real time.

Overlays:
  White crosshair  = frame centre (tracking target)
  Green box + crosshair = detected face
  Orange line = error from centre to face
  Top-left status = STARTING / SEARCHING / FACE LOCKED

Run:  ./run.sh demos/demo_face.py
Press q in the preview window or Ctrl-C to stop.
"""
import socket
import subprocess
import threading
import time

import cv2

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAM_DEV      = "/dev/video2"
CAM_W, CAM_H = 640, 360
FPS_TARGET   = 20

YAW_GAIN   = 0.90
PITCH_GAIN = 0.38
BODY_GAIN  = 0.80
HEAD_ALPHA = 0.18
BODY_ALPHA = 0.06

ANT_EXCITED =  0.70
ANT_IDLE    =  0.15
ANT_DROOP   = -0.25

LOST_TIMEOUT = 2.5

SPEAKER = "plughw:CARD=Audio,DEV=0"

# ---------------------------------------------------------------------------
# Daemon (runs in a background thread so the camera opens first)
# ---------------------------------------------------------------------------

def _launch_daemon(result: dict):
    proc = subprocess.Popen(
        ["reachy-mini-daemon", "--no-media"], start_new_session=True,
    )
    for _ in range(30):
        time.sleep(0.5)
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=0.3):
                result["proc"] = proc
                result["ready"] = True
                return
        except OSError:
            pass
    result["error"] = "Daemon did not start within 15 s"

# ---------------------------------------------------------------------------
# Sound
# ---------------------------------------------------------------------------

def found_chirp():
    subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "aevalsrc=sin(2*PI*(600+800*t)*t)*0.55:c=mono:s=22050",
        "-t", "0.14", "-f", "alsa", SPEAKER])

def lost_chirp():
    subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "aevalsrc=sin(2*PI*(900-700*t)*t)*0.45:c=mono:s=22050",
        "-t", "0.12", "-f", "alsa", SPEAKER])

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

ARM, GAP = 22, 6

def crosshair(img, px, py, color, thickness=2):
    cv2.line(img, (px - ARM, py), (px - GAP, py), color, thickness)
    cv2.line(img, (px + GAP, py), (px + ARM, py), color, thickness)
    cv2.line(img, (px, py - ARM), (px, py - GAP), color, thickness)
    cv2.line(img, (px, py + GAP), (px, py + ARM), color, thickness)
    cv2.circle(img, (px, py), GAP, color, thickness)

def draw_overlay(frame, face_box, face_seen, fps, robot_ready):
    fh, fw = frame.shape[:2]
    cx_px, cy_px = fw // 2, fh // 2

    # Centre crosshair (white, thin)
    crosshair(frame, cx_px, cy_px, (210, 210, 210), 1)

    if face_box is not None:
        x, y, w, h = face_box
        # Bounding box (mirrored coords for display)
        mx = fw - (x + w)   # mirror x for display
        cv2.rectangle(frame, (mx, y), (mx + w, y + h), (0, 220, 0), 2)
        fcx = mx + w // 2
        fcy = y + h // 2
        crosshair(frame, fcx, fcy, (0, 220, 0), 2)
        cv2.line(frame, (cx_px, cy_px), (fcx, fcy), (0, 180, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "TRACKING", (mx, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

    # Status
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Face Tracking — opening camera...")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    # Open camera immediately
    cap = cv2.VideoCapture(CAM_DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {CAM_DEV}")

    WIN = "Reachy — Face Tracking  (q to quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 450)

    # Start daemon in background thread
    daemon_result = {"ready": False, "proc": None, "error": None}
    daemon_thread = threading.Thread(target=_launch_daemon, args=(daemon_result,), daemon=True)
    daemon_thread.start()
    print("  Starting robot daemon in background...")

    # State
    target_yaw = target_pitch = target_body = 0.0
    ant_target = ANT_IDLE
    face_seen  = False
    last_seen_t = 0.0
    loop_dt    = 1.0 / FPS_TARGET
    fps_t      = time.time()
    fps_count  = fps_display = 0.0
    last_face_box = None

    mini = None
    robot_ready = False

    try:
        while True:
            t0 = time.time()

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Connect robot once daemon is ready
            if not robot_ready and daemon_result.get("ready"):
                if daemon_result.get("error"):
                    print("  ERROR:", daemon_result["error"])
                    break
                mini = ReachyMini(connection_mode="localhost_only",
                                  media_backend="no_media",
                                  spawn_daemon=False)
                mini.__enter__()
                mini.wake_up()
                robot_ready = True
                print("  Robot ready — tracking faces!")

            # Face detection
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE,
            )

            now = time.time()

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                last_face_box = (x, y, w, h)

                cx    = (x + w / 2.0) / CAM_W
                cy    = (y + h / 2.0) / CAM_H
                err_x = (cx - 0.5) * 2.0
                err_y = (cy - 0.5) * 2.0

                target_yaw   = HEAD_ALPHA * (err_x * YAW_GAIN)   + (1 - HEAD_ALPHA) * target_yaw
                target_pitch = HEAD_ALPHA * (err_y * PITCH_GAIN)  + (1 - HEAD_ALPHA) * target_pitch
                target_body  = BODY_ALPHA * (err_x * BODY_GAIN)   + (1 - BODY_ALPHA) * target_body

                ant_target  = ANT_EXCITED
                last_seen_t = now

                if not face_seen:
                    print("  ✓ Face detected!")
                    found_chirp()
                    face_seen = True

            else:
                elapsed_lost = now - last_seen_t
                if face_seen and elapsed_lost > LOST_TIMEOUT:
                    print("  ✗ Face lost — searching...")
                    lost_chirp()
                    face_seen = False
                    last_face_box = None

                if elapsed_lost > LOST_TIMEOUT:
                    target_yaw   *= 0.97
                    target_pitch *= 0.97
                    target_body  *= 0.95
                    ant_target = ANT_DROOP
                else:
                    ant_target = ANT_EXCITED if face_seen else ANT_IDLE

            # FPS
            fps_count += 1
            if now - fps_t >= 1.0:
                fps_display = fps_count / (now - fps_t)
                fps_count = 0; fps_t = now

            # Draw on a mirrored copy for display
            display = cv2.flip(frame, 1)
            draw_overlay(display, last_face_box if face_seen else None,
                         face_seen, fps_display, robot_ready)

            cv2.imshow(WIN, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Send to robot
            if robot_ready and mini:
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
        cv2.destroyAllWindows()
        cap.release()
        if mini:
            try:
                mini.__exit__(None, None, None)
            except Exception:
                pass
        proc = daemon_result.get("proc")
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        print("  Done.")


if __name__ == "__main__":
    main()
