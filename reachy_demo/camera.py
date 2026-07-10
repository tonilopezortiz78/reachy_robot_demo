"""
reachy_demo/camera.py — shared camera hub.

A single OpenCV capture thread serves MJPEG bytes and reaction callbacks to
every consumer (demo loop, face id, web preview). Latest frame is held in a
one-slot buffer so polling is O(1) and consumers never block on slow readers.

Provides:
    hub = CameraHub("/dev/video2", width=640, height=360, fps=30)
    hub.start()
    jpg = hub.mjpeg_bytes()           # bytes ready for HTTP /video multipart
    rgb = hub.frame_rgb()             # np.ndarray for face-id / vision
    hub.last_boxes                    # list[(box, name, conf, track_id)] set by face-id
    hub.overlay = drawer             # optional overlay fn(frame_bgr)->frame_bgr
    hub.stop()
"""

import os
import threading
import time
from typing import Callable

import cv2
import numpy as np

CAM_DEV = "/dev/video2"
DEFAULT_W, DEFAULT_H, DEFAULT_FPS = 640, 360, 30


class CameraHub:
    def __init__(self, dev: str = CAM_DEV, *, width=DEFAULT_W, height=DEFAULT_H, fps=DEFAULT_FPS):
        self.dev = dev
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None          # last BGR frame
        self._jpg: bytes | None = None                  # cached JPEG
        self.last_boxes: list = []                      # set by face-id
        self.overlay: Callable | None = None            # fn(frame_bgr)->frame_bgr
        self.last_fps: float = 0.0
        self.started_at: float = 0.0

    def _open_device(self, dev: str) -> "cv2.VideoCapture | None":
        """Open+configure a V4L2 capture. Returns None if it won't open/read."""
        try:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            pass
        return None

    def _pick_device(self) -> "cv2.VideoCapture | None":
        """Find a working capture device. Prefer the configured node, but after a
        USB re-enumeration (the dance can jostle the cable) the camera may come back
        on a different /dev/videoN, so fall back to probing the even-numbered nodes
        (UVC capture nodes; odd ones are usually metadata)."""
        candidates = [self.dev] + [f"/dev/video{i}" for i in (2, 0, 4, 6, 1, 3)]
        seen = set()
        for dev in candidates:
            if dev in seen or not os.path.exists(dev):
                continue
            seen.add(dev)
            cap = self._open_device(dev)
            if cap is not None:
                ok, _ = cap.read()          # a node that opens but never reads is useless
                if ok:
                    self.dev = dev
                    return cap
                cap.release()
        return None

    def start(self):
        self._cap = self._open_device(self.dev)
        if self._cap is None:
            raise RuntimeError(f"camera: cannot open {self.dev}")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.started_at = time.time()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def _loop(self):
        fps_t = time.time()
        fps_count = 0
        fail = 0
        while not self._stop.is_set():
            cap = self._cap
            ok, frame = False, None
            if cap is not None:
                try:
                    ok, frame = cap.read()
                except Exception:
                    ok = False
            if not ok:
                fail += 1
                # ~1 s of failed reads → the USB link dropped (a vigorous dance can
                # jostle the cable). The old handle returns False forever even after
                # the device re-enumerates, so release it and reopen a fresh capture
                # (possibly on a new /dev/videoN). Backoff so a truly-unplugged
                # camera doesn't spin the CPU.
                if fail >= 25:
                    try:
                        if self._cap is not None:
                            self._cap.release()
                    except Exception:
                        pass
                    self._cap = None
                    newcap = self._pick_device()
                    if newcap is not None:
                        self._cap = newcap
                        fail = 0
                        print(f"  [camera] reconnected on {self.dev}", flush=True)
                    else:
                        time.sleep(1.0)   # camera still gone — wait before retrying
                    continue
                time.sleep(0.02)
                continue
            fail = 0
            # Store the CLEAN frame — overlay (boxes/name labels) is applied
            # only in mjpeg_bytes() for the dashboard. Drawing into the stored
            # frame contaminated frame_rgb()/frame_bgr(), so face recognition
            # and enrollment saw the previous detection's boxes painted over
            # the faces.
            with self._lock:
                self._frame = frame
                self._jpg = None
            fps_count += 1
            if time.time() - fps_t >= 1.0:
                self.last_fps = fps_count / (time.time() - fps_t)
                fps_count = 0
                fps_t = time.time()

    def frame_bgr(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def frame_rgb(self) -> np.ndarray | None:
        f = self.frame_bgr()
        if f is None:
            return None
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

    def mjpeg_bytes(self) -> bytes | None:
        with self._lock:
            f = self._frame
            j = self._jpg
        if f is None:
            return None
        if j is not None:
            return j
        if self.overlay is not None:
            # Draw boxes on a COPY so the stored frame stays clean for
            # recognition consumers (frame_rgb/frame_bgr).
            try:
                f = self.overlay(f.copy(), self.last_boxes)
            except Exception:
                pass
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return None
        jpg = bytes(buf.tobytes())
        with self._lock:
            if self._jpg is None:
                self._jpg = jpg
        return jpg