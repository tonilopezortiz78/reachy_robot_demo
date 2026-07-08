"""
reachy_demo/recorder.py — rolling "black box" diagnostic recorder.

DiagnosticRecorder continuously captures everything the robot experienced —
console/event text, microphone audio clips, and downsampled camera video —
into a ROLLING window capped at a fixed disk budget (default ~100 MB), so a
bad interaction can be diagnosed after the fact without needing to reproduce
it live and without the recordings ever filling the disk.

Layout, all under `<base_dir>/diag/`:

    diag/
      events.log            — timestamped text log of everything log_text()'d
      events.log.1          — previous events.log, rotated when it exceeds
                              ~10 MB (so text alone can never grow unbounded)
      video/seg_000001.mp4  — ~segment_seconds chunks of downsampled camera
      video/seg_000002.mp4    frames, rolled over continuously
      audio/utt_000001.wav  — each clip passed to add_audio(), 16kHz mono s16
      audio/utt_000002.wav

Budget enforcement: after every finished video segment and every saved audio
clip, the recorder sums the size of everything under `diag/` and, while it
exceeds `budget_mb`, deletes the OLDEST files first (by mtime) until it's back
under budget. The file currently being written (the open video segment) is
always protected from deletion. events.log/.1 are excluded from this
oldest-first sweep — they're capped separately by their own rotation so they
never compete with video/audio for the prune, but their size still counts
toward the total budget.

Design goals:
  - add_frame() is NON-BLOCKING: frames are pushed onto a small bounded queue
    and DROPPED if the queue is full, so a slow disk or codec never stalls the
    camera thread. A single background thread drains the queue, throttles to
    `fps` (skipping frames that arrive faster), and writes to the current
    cv2.VideoWriter.
  - Never raises into the caller. This is a diagnostic aid, not part of the
    control path — any failure (codec missing, disk full, VideoWriter refusing
    to open, ...) is caught, logged once, and degrades gracefully (e.g. video
    recording disables itself while audio + text logging keep working).

Typical usage:

    rec = DiagnosticRecorder(Path("."), budget_mb=100, fps=10)
    rec.start()
    ...
    rec.add_frame(camera_hub.frame_rgb())   # called from the camera loop
    rec.add_audio(pcm, tag="utt")           # called when an utterance is captured
    rec.log_text("visitor greeted: Alice")  # called from anywhere
    ...
    rec.stop()                              # flushes + closes the last segment
"""

import queue
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Cap on events.log before it's rotated to events.log.1 — keeps the text log
# itself from ever growing unbounded, independent of the video/audio budget.
_EVENTS_LOG_MAX_BYTES = 10 * 1024 * 1024


class DiagnosticRecorder:
    """Rolling black-box recorder: text + audio + video, capped at `budget_mb`.

    All public methods are safe to call from any thread and never raise.
    """

    def __init__(
        self,
        base_dir,
        budget_mb: int = 100,
        fps: int = 10,
        frame_size: Tuple[int, int] = (640, 360),
        segment_seconds: int = 20,
    ):
        self.base_dir = Path(base_dir)
        self.diag_dir = self.base_dir / "diag"
        self.video_dir = self.diag_dir / "video"
        self.audio_dir = self.diag_dir / "audio"
        self.events_path = self.diag_dir / "events.log"

        try:
            self.video_dir.mkdir(parents=True, exist_ok=True)
            self.audio_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self.budget_bytes = max(1, budget_mb) * 1024 * 1024
        self.fps = max(1, fps)
        self.frame_size = frame_size
        self.segment_seconds = max(1, segment_seconds)

        # threading / queueing
        self._lock = threading.RLock()          # RLock: prune is called from
                                                  # nested contexts (segment
                                                  # close -> prune) on the same
                                                  # thread.
        self._stop_event = threading.Event()
        self._frame_q: "queue.Queue" = queue.Queue(maxsize=4)
        self._video_thread: Optional[threading.Thread] = None
        self._started = False

        # video state (guarded by self._lock)
        self._writer = None                      # cv2.VideoWriter | None
        self._writer_path: Optional[Path] = None
        self._seg_counter = 0
        self._current_segment_start = 0.0
        self._video_enabled = True
        self._video_disabled_logged = False

        # audio state
        self._audio_counter = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        """Start the background video-writer thread. Safe to call once."""
        try:
            with self._lock:
                if self._started:
                    return
                self._stop_event.clear()
                self._video_thread = threading.Thread(
                    target=self._video_loop, daemon=True, name="DiagRecorderVideo"
                )
                self._video_thread.start()
                self._started = True
            self.log_text("[recorder] started")
        except Exception:
            pass

    def stop(self):
        """Flush + close the current video segment and join background threads."""
        try:
            self.log_text("[recorder] stopping")
        except Exception:
            pass
        try:
            self._stop_event.set()
            t = self._video_thread
            if t is not None:
                t.join(timeout=max(5.0, self.segment_seconds + 2.0))
            with self._lock:
                self._close_writer()
                self._started = False
                self._video_thread = None
        except Exception:
            pass

    # ── text log ───────────────────────────────────────────────────────────

    def log_text(self, msg: str):
        """Append a timestamped line to diag/events.log. Never raises."""
        try:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            line = f"{stamp} {msg}\n"
            with self._lock:
                self._rotate_events_log_if_needed()
                with open(self.events_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            pass

    def _rotate_events_log_if_needed(self):
        """If events.log has grown past the cap, rename it to events.log.1
        and start a fresh one. Keeps the text log itself bounded regardless
        of the video/audio budget."""
        try:
            if not self.events_path.exists():
                return
            if self.events_path.stat().st_size <= _EVENTS_LOG_MAX_BYTES:
                return
            backup = self.diag_dir / "events.log.1"
            try:
                if backup.exists():
                    backup.unlink()
            except OSError:
                pass
            self.events_path.rename(backup)
        except Exception:
            pass

    # ── audio ──────────────────────────────────────────────────────────────

    def add_audio(self, pcm: bytes, tag: str = "utt"):
        """Save `pcm` (int16 mono @ 16kHz) as diag/audio/<tag>_<NNNNNN>.wav."""
        try:
            with self._lock:
                self._audio_counter += 1
                idx = self._audio_counter
            safe_tag = "".join(c for c in tag if c.isalnum() or c in ("-", "_")) or "utt"
            path = self.audio_dir / f"{safe_tag}_{idx:06d}.wav"
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm)
        except Exception:
            return
        self._prune_if_needed()

    # ── video ──────────────────────────────────────────────────────────────

    def add_frame(self, rgb: "np.ndarray"):
        """Feed a camera frame (RGB np array). NON-BLOCKING: dropped if the
        internal queue is full, so a slow writer never stalls the caller."""
        if not self._started or not self._video_enabled or rgb is None:
            return
        try:
            self._frame_q.put_nowait(rgb)
        except queue.Full:
            pass  # drop — by design, never block the camera thread
        except Exception:
            pass

    def _video_loop(self):
        """Background thread: drains the frame queue, throttles to `fps`,
        writes to the current segment, and rolls segments over time."""
        interval = 1.0 / self.fps
        last_write = 0.0
        while not self._stop_event.is_set():
            try:
                frame = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not self._video_enabled:
                continue
            now = time.monotonic()
            if now - last_write < interval:
                continue  # throttle: arrived faster than fps, drop it
            last_write = now
            self._write_frame(frame)
        with self._lock:
            self._close_writer()

    def _write_frame(self, rgb):
        try:
            with self._lock:
                self._maybe_roll_segment()
                if self._writer is None:
                    return
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                resized = cv2.resize(bgr, self.frame_size)
                self._writer.write(resized)
        except Exception as e:
            self.log_text(f"[recorder] frame write failed: {e}")

    def _maybe_roll_segment(self):
        """Must be called with self._lock held. Opens the first segment, or
        rolls to a new one once segment_seconds has elapsed."""
        if not self._video_enabled:
            return
        if (
            self._writer is None
            or (time.monotonic() - self._current_segment_start) >= self.segment_seconds
        ):
            self._close_writer()
            self._open_new_segment()

    def _open_new_segment(self):
        """Must be called with self._lock held."""
        if not self._video_enabled:
            return
        self._seg_counter += 1
        path = self.video_dir / f"seg_{self._seg_counter:06d}.mp4"
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w, h = self.frame_size
            writer = cv2.VideoWriter(str(path), fourcc, float(self.fps), (int(w), int(h)))
            if not writer.isOpened():
                writer.release()
                raise RuntimeError("cv2.VideoWriter failed to open (missing codec?)")
            self._writer = writer
            self._writer_path = path
            self._current_segment_start = time.monotonic()
        except Exception as e:
            self._writer = None
            self._writer_path = None
            self._video_enabled = False
            if not self._video_disabled_logged:
                self._video_disabled_logged = True
                self.log_text(
                    f"[recorder] video recording disabled (audio+text logging "
                    f"continue): {e}"
                )

    def _close_writer(self):
        """Must be called with self._lock held. Flushes+closes the current
        segment (if any) so it's playable, then prunes to budget."""
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None
            self._writer_path = None
            self._prune_if_needed()

    # ── rolling budget enforcement ───────────────────────────────────────────

    def _dir_size_and_candidates(self):
        """Walk diag/ once: total size of everything, plus the list of files
        eligible for deletion (everything except events.log/.1 and whatever
        segment is currently being written)."""
        protect = {self.events_path, self.diag_dir / "events.log.1"}
        if self._writer_path is not None:
            protect.add(self._writer_path)
        total = 0
        candidates = []
        try:
            for p in self.diag_dir.rglob("*"):
                if not p.is_file():
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                total += st.st_size
                if p not in protect:
                    candidates.append((p, st.st_size, st.st_mtime))
        except Exception:
            pass
        return total, candidates

    def _prune_if_needed(self):
        """While diag/ exceeds the budget, delete the oldest eligible files
        (by mtime) first. Never deletes events.log/.1 or the segment
        currently being written."""
        try:
            with self._lock:
                total, candidates = self._dir_size_and_candidates()
                if total <= self.budget_bytes:
                    return
                candidates.sort(key=lambda t: t[2])  # oldest mtime first
                for path, size, _mtime in candidates:
                    if total <= self.budget_bytes:
                        break
                    try:
                        path.unlink()
                        total -= size
                    except OSError:
                        continue
        except Exception:
            pass
