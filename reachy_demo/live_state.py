"""
reachy_demo/live_state.py — live robot state bridge.

A single shared mutable object that the demo loop writes to and the web
dashboard reads from. No locks — every field is a primitive or simple list/dict
that Python assigns atomically. The web server polls this for the JSON status
endpoint and for overlay drawing on the camera feed.

Threads: written by the demo main loop + listener thread; read by the
FastAPI server thread. All fields are safe to read at any time even during a
write — worst case the JSON shows one frame of stale data, which is fine for
a status dashboard.
"""

import time
from dataclasses import dataclass, field


@dataclass
class LiveState:
    # Robot
    robot_online: bool = False
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    body_yaw: float = 0.0
    antenna_left: float = 0.0
    antenna_right: float = 0.0

    # Conversation
    anim_state: str = "idle"          # idle | listening | thinking | speaking
    current_lang: str = "—"
    last_user: str = ""
    last_reply: str = ""
    llm_partial: str = ""             # in-progress LLM token stream (stage "thinking")
    current_gesture: str = ""         # gesture name currently being played, e.g. "celebrate"
    kid_mode: bool = True             # kid-mode layer on/off (control panel can toggle)
    last_face_name: str = "—"
    last_face_conf: float = 0.0
    turn_count: int = 0
    uptime_s: float = 0.0
    started_at: float = field(default_factory=time.time)

    # Latency tracking (last turn, seconds)
    stt_s: float = 0.0
    llm_ttf_s: float = 0.0
    tts_tta_s: float = 0.0
    total_s: float = 0.0

    # Faces
    faces_visible: int = 0
    known_person_count: int = 0        # roster size

    # LLM provider in use
    llm_provider: str = "groq"         # groq | cerebras
    llm_model: str = ""                # e.g. "gemma-4-31b" (shown only in the Costs tab)
    tokens_in: int = 0                 # cumulative estimated input tokens this session
    tokens_out: int = 0                # cumulative estimated output tokens this session
    est_cost_usd: float = 0.0          # cumulative estimated cost, USD

    # Current known speaker
    person_summary: str = ""           # short profile of the current known speaker

    # Floor control
    muted: bool = False
    volume: float = 2.5              # ffmpeg volume multiplier (1.0=unity, 2.5=+8dB)
    speech_rate: str = "+20%"        # edge-tts rate offset
    energy: float = 1.0              # antenna liveliness 0.0-1.0

    # Manual control requests from web UI (demo loop drains these)
    pending_wake: bool = False
    pending_sleep: bool = False
    pending_say: str = ""
    pending_shutdown: bool = False   # full demo stop: robot to sleep, process exits
    pending_gesture: str = ""        # operator-triggered gesture name (control panel)
    pending_dance: bool = False      # operator-triggered Macarena (control panel)
    pending_dance_name: str = ""     # which dance: "macarena"|"robot_wave"|"happy_hop"

    def snapshot(self) -> dict:
        u = time.time() - self.started_at
        return {
            "robot_online": self.robot_online,
            "head_yaw": round(self.head_yaw, 3),
            "head_pitch": round(self.head_pitch, 3),
            "body_yaw": round(self.body_yaw, 3),
            "antenna_left": round(self.antenna_left, 3),
            "antenna_right": round(self.antenna_right, 3),
            "anim_state": self.anim_state,
            "current_lang": self.current_lang,
            "last_user": self.last_user[:200],
            "last_reply": self.last_reply[:200],
            "llm_partial": self.llm_partial[:300],
            "current_gesture": self.current_gesture,
            "kid_mode": self.kid_mode,
            "last_face_name": self.last_face_name,
            "last_face_conf": round(self.last_face_conf, 3),
            "turn_count": self.turn_count,
            "uptime_s": round(u, 1),
            "stt_s": round(self.stt_s, 3),
            "llm_ttf_s": round(self.llm_ttf_s, 3),
            "tts_tta_s": round(self.tts_tta_s, 3),
            "total_s": round(self.total_s, 3),
            "faces_visible": self.faces_visible,
            "known_person_count": self.known_person_count,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "est_cost_usd": round(self.est_cost_usd, 6),
            "person_summary": self.person_summary[:400],
            "muted": self.muted,
            "volume": self.volume,
            "speech_rate": self.speech_rate,
            "energy": self.energy,
        }

    def request_wake(self):
        self.pending_wake = True

    def request_sleep(self):
        self.pending_sleep = True

    def request_say(self, text: str):
        self.pending_say = text.strip()[:200]

    def request_gesture(self, name: str):
        self.pending_gesture = name.strip()[:40]

    def request_dance(self, name="macarena"):
        self.pending_dance = True
        self.pending_dance_name = name[:20]

    def request_shutdown(self):
        self.pending_shutdown = True