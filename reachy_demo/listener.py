"""
reachy_demo/listener.py — continuous VAD microphone listener.

Single source of truth for the background mic listener used by the talking
demos (previously copy-pasted into each demo, which let them drift). Opens a
pacat capture on the robot mic, runs Silero VAD continuously on a background
thread, and posts utterance events to a queue:

    {"type": "start"}                 — user started speaking
    {"type": "end", "pcm": bytes}     — user stopped; full utterance PCM attached
    {"type": "mic_error", "reason"}   — mic lost and could not be recovered

Two robustness features baked in (both learned the hard way on this hardware):
  • Threshold modes — "normal" while the robot is silent, "barge_in" (higher,
    needs sustained speech) while the robot is talking, so its own speaker
    bleed doesn't constantly trip the VAD.
  • Auto-recovery — if the capture stream dies (USB replug, PipeWire restart,
    suspended source), it RE-DETECTS the robot mic and reopens, retrying a few
    times before giving up. This is the "robot stops listening after I touch
    the audio settings" fix. See reachy_demo.audio.redetect_mic.
"""
import subprocess
import threading
import time

import numpy as np
import torch
from silero_vad import VADIterator

import reachy_demo.audio as audio   # live audio.MIC (updated by redetect_mic)
from reachy_demo.audio import MIC_RATE, VAD_CHUNK, cleanup_orphan_capture, redetect_mic

# ── Default VAD tuning (matches every demo's previous local constants) ─────────
THRESH_NORMAL   = 0.45   # standard — when robot is silent
THRESH_BARGE_IN = 0.75   # high — when robot is speaking, only real speech counts
SILENCE_MS      = 700    # silence before an utterance is considered ended
MIN_SPEECH_S    = 0.30   # shorter than this is dropped (cough / click)
TAIL_FRAMES     = 10     # extra frames kept after end so words aren't clipped
BARGE_IN_FRAMES = 6      # ~200 ms of sustained high-threshold speech to barge in
MAX_RECOVER     = 5      # consecutive failed reopens before a hard mic_error


class ContinuousListener:
    """Background thread: opens pacat, runs VAD continuously, posts events.

    VAD tuning can be overridden per-instance but defaults match the values the
    demos used. Pass `log` (a SessionLogger) to get capture/recovery breadcrumbs.
    """

    def __init__(self, vad_model, event_queue, log=None, *,
                 thresh_normal=THRESH_NORMAL, thresh_barge_in=THRESH_BARGE_IN,
                 silence_ms=SILENCE_MS, min_speech_s=MIN_SPEECH_S,
                 tail_frames=TAIL_FRAMES, barge_in_frames=BARGE_IN_FRAMES,
                 max_recover=MAX_RECOVER):
        self.vad_model = vad_model
        self.q = event_queue
        self.log = log
        self._thresh_normal = thresh_normal
        self._thresh_barge_in = thresh_barge_in
        self._silence_ms = silence_ms
        self._min_speech_s = min_speech_s
        self._tail_frames = tail_frames
        self._barge_in_frames = barge_in_frames
        self._max_recover = max_recover
        self._stop = threading.Event()
        self._muted = False
        self._threshold_mode = "normal"
        self._consecutive_triggers = 0
        self._in_speech = False
        self._ended = False
        self._tail_count = 0
        self._speech_buf = []
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def mute(self):
        """Discard mic input (used while the robot plays a cue, so it never
        captures its own voice through speaker→mic bleed)."""
        self._muted = True

    def unmute(self):
        self._muted = False

    def set_threshold_mode(self, mode: str):
        assert mode in ("normal", "barge_in")
        self._threshold_mode = mode
        if mode == "barge_in" and not self._in_speech:
            self._consecutive_triggers = 0

    def _current_threshold(self) -> float:
        return (self._thresh_barge_in if self._threshold_mode == "barge_in"
                else self._thresh_normal)

    def _open_capture(self):
        """Open a fresh pacat capture on the live, re-detected robot mic.
        Reading audio.MIC live (not a frozen import) means a replug / PipeWire
        restart that changed the source name is picked up on reopen."""
        device = audio.MIC
        return subprocess.Popen(
            ["pacat", "--record", "--raw",
             f"--device={device}",
             f"--rate={MIC_RATE}", "--channels=1", "--format=s16le"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ), device

    def _loop(self):
        arecord, device = self._open_capture()
        if self.log:
            self.log.event(f"  [listener] capturing on {device}")
        vad_iter = None
        recover_attempts = 0   # consecutive reopen attempts after a stream death
        try:
            while not self._stop.is_set():
                if vad_iter is None:
                    vad_iter = VADIterator(
                        self.vad_model, sampling_rate=MIC_RATE,
                        threshold=self._current_threshold(),
                        min_silence_duration_ms=self._silence_ms,
                    )
                    self._consecutive_triggers = 0
                    self._in_speech = False
                    self._ended = False
                    self._tail_count = 0
                    self._speech_buf = []

                raw = arecord.stdout.read(VAD_CHUNK * 2)
                if not raw or len(raw) < VAD_CHUNK * 2:
                    # Mic stream died — device lost, suspended, or grabbed by
                    # another process (e.g. cable replug, PipeWire restart).
                    # Instead of giving up forever, RE-DETECT the robot mic and
                    # reopen. Only surface a hard error if several reopen attempts
                    # in a row fail.
                    if self.log:
                        self.log.event(
                            f"  [listener] stream closed (got "
                            f"{len(raw) if raw else 0} bytes) — recovering...")
                    try:
                        arecord.terminate(); arecord.wait(timeout=1.0)
                    except Exception:
                        pass
                    recover_attempts += 1
                    if recover_attempts > self._max_recover:
                        self.q.put({"type": "mic_error",
                                    "reason": (f"mic stream kept closing after "
                                               f"{self._max_recover} reopen "
                                               "attempts — device lost")})
                        break
                    cleanup_orphan_capture()
                    new_dev = redetect_mic()        # re-poll PipeWire for the robot mic
                    time.sleep(0.4)                 # let the device settle
                    arecord, device = self._open_capture()
                    if self.log and new_dev != device:
                        self.log.event(f"  [listener] re-detected mic: {new_dev}")
                    vad_iter = None
                    continue
                recover_attempts = 0   # got a good frame — reset the recovery counter

                if self._muted:
                    # Robot is speaking a cue — discard this audio and reset VAD
                    # state so the cue is never mistaken for the user talking.
                    vad_iter = None
                    continue

                if vad_iter.threshold != self._current_threshold():
                    vad_iter = None
                    continue

                audio_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                result = vad_iter(torch.from_numpy(audio_f32))

                if self._threshold_mode == "barge_in" and not self._in_speech:
                    if result and "start" in result:
                        self._consecutive_triggers += 1
                        if self._consecutive_triggers >= self._barge_in_frames:
                            self._in_speech = True
                            self._speech_buf = [raw]
                            self.q.put({"type": "start"})
                    else:
                        self._consecutive_triggers = max(0, self._consecutive_triggers - 1)
                else:
                    if result and "start" in result and not self._in_speech:
                        self._in_speech = True
                        self._speech_buf = [raw]
                        self.q.put({"type": "start"})

                if self._in_speech:
                    self._speech_buf.append(raw)

                if result and "end" in result and self._in_speech and not self._ended:
                    self._ended = True

                if self._ended:
                    self._tail_count += 1
                    if self._tail_count >= self._tail_frames:
                        min_frames = int(self._min_speech_s * MIC_RATE / VAD_CHUNK)
                        if len(self._speech_buf) >= min_frames:
                            self.q.put({"type": "end", "pcm": b"".join(self._speech_buf)})
                        self._in_speech = False
                        self._ended = False
                        self._tail_count = 0
                        self._speech_buf = []
                        self._consecutive_triggers = 0
        finally:
            arecord.terminate()
            arecord.wait()
