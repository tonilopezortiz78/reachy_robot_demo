"""
reachy_demo/web_server.py — FastAPI web dashboard for the Reachy Mini robot.

Serves a single-page local website with a live MJPEG camera feed and basic
robot info/controls. The dashboard reads from shared objects (LiveState and
CameraHub) that the main demo owns and passes in — it never instantiates them
itself and never touches the robot directly. Control buttons simply set flags
on the LiveState that the main demo loop drains.

Endpoints:
    GET  /          — full HTML dashboard (single page, inline CSS/JS)
    GET  /video     — MJPEG live stream (multipart/x-mixed-replace; boundary=frame)
    GET  /status    — JSON snapshot of LiveState (kept for external tools; the
                      page only polls it as a fallback when the WebSocket is down)
    WS   /ws        — pushes the LiveState snapshot as JSON text frames ~8-10x/s,
                      but only when the snapshot changed since the last push
    POST /api/wake  — sets state.pending_wake  -> {"ok": true}
    POST /api/sleep — sets state.pending_sleep -> {"ok": true}
    POST /api/say   — JSON {text: "..."} -> state.request_say -> {"ok": true}
    POST /api/mute  — JSON {muted: bool} -> sets state.muted -> {"ok": true}
    POST /api/stop  — sets state.pending_shutdown (demo exits cleanly, robot sleeps)

Threading model:
    start() spawns uvicorn in a daemon thread and returns immediately so the
    caller's main loop keeps running. stop() just sets a flag; the daemon
    thread (and the MJPEG generator) dies with the process — no graceful
    uvicorn shutdown is attempted.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from starlette.responses import HTMLResponse, StreamingResponse

if TYPE_CHECKING:
    from reachy_demo.camera import CameraHub
    from reachy_demo.live_state import LiveState


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reachy Mini &mdash; Control Dashboard</title>
<style>
:root {
  --bg: #0d1117;
  --card: #161b22;
  --text: #c9d1d9;
  --accent: #58a6ff;
  --green: #3fb950;
  --red: #f85149;
  --orange: #d29922;
  --purple: #bc8cff;
  --border: #30363d;
  --muted: #8b949e;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  min-height: 100vh;
  padding: 20px;
  transition: background 0.6s ease;
}
/* Ambient background tint keyed to the robot's anim_state (set via JS as
   body[data-state], driven by the /ws WebSocket push, or /status polling
   as fallback). Falls back to the default --bg above when data-state is
   absent/unrecognised. Kept dark and slightly desaturated so card text
   stays readable in every state. */
body[data-state="idle"]      { background: #0f172a; }
body[data-state="listening"] { background: #14532d; }
body[data-state="thinking"]  { background: #78350f; }
body[data-state="speaking"]  { background: #1e3a5f; }
header { text-align: center; margin-bottom: 24px; }
header h1 { font-size: 1.5rem; font-weight: 600; }
header .subtitle {
  color: var(--muted);
  font-size: 0.85rem;
  font-family: "SF Mono", "Cascadia Code", "Fira Code", monospace;
  margin-top: 4px;
}
.grid {
  display: grid;
  grid-template-columns: 1fr 380px;
  gap: 20px;
  max-width: 1400px;
  margin: 0 auto;
}
@media (max-width: 900px) {
  .grid { grid-template-columns: 1fr; }
}
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  margin-bottom: 16px;
}
.card h2 {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  margin-bottom: 12px;
}
.video-wrap {
  position: relative;
  background: #000;
  border-radius: 10px;
  overflow: hidden;
}
.video-wrap img {
  width: 100%;
  display: block;
  border-radius: 10px;
  box-shadow: 0 0 24px rgba(88, 166, 255, 0.12);
}
.rec {
  display: none;
  position: absolute;
  top: 12px;
  left: 14px;
  color: var(--red);
  font-size: 14px;
  font-weight: bold;
  font-family: "SF Mono", "Cascadia Code", monospace;
  text-shadow: 0 0 6px rgba(0, 0, 0, 0.9);
  letter-spacing: 0.05em;
}
.rec.active { display: block; animation: pulse 1s ease-in-out infinite; }
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.15; }
}
.status-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
  gap: 12px;
}
.status-row:last-child { border-bottom: none; }
.status-label { color: var(--muted); font-size: 0.85rem; flex-shrink: 0; }
.status-value {
  font-size: 0.9rem;
  text-align: right;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.status-value-wrap {
  font-size: 0.9rem;
  text-align: right;
  word-break: break-word;
  white-space: normal;
  line-height: 1.4;
}
.mono {
  font-family: "SF Mono", "Cascadia Code", "Fira Code", "Consolas", monospace;
}
.dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 6px;
  vertical-align: middle;
}
.dot-on { background: var(--green); box-shadow: 0 0 8px var(--green); }
.dot-off { background: var(--red); box-shadow: 0 0 8px var(--red); }
.badge {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 12px;
  font-size: 0.78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.badge-idle { background: #21262d; color: var(--muted); border: 1px solid var(--border); }
.badge-listening { background: rgba(88,166,255,0.15); color: var(--accent); border: 1px solid var(--accent); }
.badge-thinking { background: rgba(210,153,34,0.15); color: var(--orange); border: 1px solid var(--orange); }
.badge-speaking { background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid var(--green); }
.chip {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 4px;
  font-size: 0.78rem;
  font-weight: 600;
  font-family: "SF Mono", "Cascadia Code", monospace;
}
.chip-groq { background: rgba(210,153,34,0.15); color: var(--orange); border: 1px solid var(--orange); }
.chip-cerebras { background: rgba(188,140,255,0.15); color: var(--purple); border: 1px solid var(--purple); }
.conf { color: var(--muted); font-size: 0.8rem; font-family: "SF Mono", "Cascadia Code", monospace; }
.person-summary {
  font-size: 0.9rem;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}
.latency-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin-bottom: 16px;
}
.lat-card { text-align: center; padding: 12px 6px; }
.lat-label {
  display: block;
  font-size: 0.68rem;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 4px;
  letter-spacing: 0.03em;
}
.lat-val {
  display: block;
  font-family: "SF Mono", "Cascadia Code", "Fira Code", monospace;
  font-size: 1rem;
  color: var(--accent);
}
.controls { display: flex; flex-direction: column; gap: 12px; }
.btn-row { display: flex; gap: 8px; flex-wrap: wrap; }
button {
  background: #21262d;
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 18px;
  font-size: 0.85rem;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s, color 0.15s, transform 0.05s;
}
button:hover { background: #30363d; border-color: var(--accent); color: var(--accent); }
button:active { transform: scale(0.97); }
.say-row { display: flex; gap: 8px; }
.say-row input {
  flex: 1;
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 0.85rem;
}
.say-row input:focus { outline: none; border-color: var(--accent); }
.say-row button { flex-shrink: 0; }
.conn-banner {
  display: none;
  background: rgba(248,81,73,0.15);
  border: 1px solid var(--red);
  color: var(--red);
  text-align: center;
  padding: 8px 16px;
  border-radius: 8px;
  margin-bottom: 16px;
  font-size: 0.85rem;
  font-weight: 600;
  max-width: 1400px;
  margin-left: auto;
  margin-right: auto;
  margin-bottom: 16px;
}
.conn-banner.show { display: block; animation: pulse 1.5s ease-in-out infinite; }
.video-overlay {
  display: none;
  position: absolute;
  top: 50%; left: 50%;
  transform: translate(-50%,-50%);
  color: var(--muted);
  font-size: 0.9rem;
  font-family: "SF Mono","Cascadia Code",monospace;
  background: rgba(0,0,0,0.8);
  padding: 10px 20px;
  border-radius: 8px;
  pointer-events: none;
}
.video-overlay.show { display: block; }
.tabs { display: flex; gap: 8px; margin-bottom: 12px; }
.tab-btn {
  padding: 6px 16px;
  border-radius: 8px;
  border: 1px solid rgba(127,127,127,0.35);
  background: transparent;
  color: inherit;
  cursor: pointer;
  font-size: 0.85rem;
}
.tab-btn.active { background: rgba(127,127,127,0.25); font-weight: 600; }
.tabpane { display: none; }
.tabpane.active { display: block; }
.btn-danger {
  background: #b3362b !important;
  color: #fff !important;
  border-color: #b3362b !important;
}
/* Big human-readable state banner: tells visitors when THEY can talk and
   when Reachy is thinking/talking. Colored in sync with the body tint. */
.state-banner {
  text-align: center;
  font-size: 2.2rem;
  font-weight: 700;
  padding: 14px 10px;
  border-radius: 12px;
  margin: 0 24px 14px;
  letter-spacing: 0.02em;
  transition: background 0.3s, color 0.3s;
}
body[data-state="idle"]      .state-banner { background: #1e293b; color: #94a3b8; }
body[data-state="listening"] .state-banner { background: #16a34a; color: #f0fdf4; }
body[data-state="thinking"]  .state-banner { background: #d97706; color: #fffbeb; }
body[data-state="speaking"]  .state-banner { background: #2563eb; color: #eff6ff; }
.person-block { margin-bottom: 14px; }
.person-block h3 { margin: 0 0 6px; font-size: 1rem; }
.person-block ul { margin: 0; padding-left: 20px; font-size: 0.88rem; }
.person-block li { margin-bottom: 3px; }
</style>
</head>
<body>
<header>
  <h1>Reachy Mini &mdash; Control Dashboard</h1>
  <p class="subtitle">localhost:__PORT__</p>
</header>
<div class="conn-banner" id="conn-banner">&#9888; Connection lost &mdash; reconnecting<span id="conn-dots"></span></div>
<div class="state-banner" id="state-banner">&#128564; Reachy is idle</div>
<main class="grid">
   <div class="left">
     <div class="card">
       <div class="video-wrap">
        <img id="video-feed" src="/video" alt="Live camera feed" onerror="videoError()" onload="videoOK()">
        <div class="video-overlay" id="video-overlay">Camera reconnecting<span id="vid-dots"></span></div>
         <div class="rec" id="rec">&#9679; REC</div>
       </div>
     </div>
   </div>
  <div class="right">
    <div class="tabs">
      <button class="tab-btn active" id="tab-btn-robot" onclick="showTab('robot')">Robot</button>
      <button class="tab-btn" id="tab-btn-people" onclick="showTab('people')">People</button>
      <button class="tab-btn" id="tab-btn-admin" onclick="showTab('admin')">Admin</button>
    </div>
    <div class="tabpane active" id="tab-robot">
    <div class="card">
      <h2>Status</h2>
      <div class="status-row">
        <span class="status-label">Robot Online</span>
        <span class="status-value"><span class="dot dot-off" id="robot-dot"></span><span id="robot-status">Offline</span></span>
      </div>
      <div class="status-row">
        <span class="status-label">Animation</span>
        <span class="status-value"><span class="badge badge-idle" id="anim-badge">idle</span></span>
      </div>
      <div class="status-row">
        <span class="status-label">Language</span>
        <span class="status-value" id="current-lang">&mdash;</span>
      </div>
      <div class="status-row">
        <span class="status-label">Last User</span>
        <span class="status-value-wrap" id="last-user">&mdash;</span>
      </div>
      <div class="status-row">
        <span class="status-label">Last Reply</span>
        <span class="status-value-wrap" id="last-reply">&mdash;</span>
      </div>
      <div class="status-row">
        <span class="status-label">Last Face</span>
        <span class="status-value"><span id="last-face-name">&mdash;</span><span class="conf" id="last-face-conf"></span></span>
      </div>
      <div class="status-row">
        <span class="status-label">Turn Count</span>
        <span class="status-value mono" id="turn-count">0</span>
      </div>
      <div class="status-row">
        <span class="status-label">Uptime</span>
        <span class="status-value mono" id="uptime">00:00</span>
      </div>
      <div class="status-row">
        <span class="status-label">Faces Visible</span>
        <span class="status-value mono" id="faces-visible">0</span>
      </div>
      <div class="status-row">
        <span class="status-label">Known Persons</span>
        <span class="status-value mono" id="known-person-count">0</span>
      </div>
      <div class="status-row">
        <span class="status-label">Mic</span>
        <span class="status-value" id="mic-status">Unmuted</span>
      </div>
    </div>
    <div class="latency-row">
      <div class="card lat-card"><span class="lat-label">STT</span><span class="lat-val" id="lat-stt">0.00s</span></div>
      <div class="card lat-card"><span class="lat-label">LLM TTF</span><span class="lat-val" id="lat-llm">0.00s</span></div>
      <div class="card lat-card"><span class="lat-label">TTS TTA</span><span class="lat-val" id="lat-tts">0.00s</span></div>
      <div class="card lat-card"><span class="lat-label">Total</span><span class="lat-val" id="lat-total">0.00s</span></div>
    </div>
    <div class="card">
      <h2>Person</h2>
      <div class="person-summary" id="person-summary">&mdash;</div>
    </div>
    <div class="card">
      <h2>Controls</h2>
      <div class="controls">
        <div class="btn-row">
          <button id="btn-wake">Wake Up</button>
          <button id="btn-sleep">Go Sleep</button>
          <button id="btn-mute">Mute</button>
          <button id="btn-stop" class="btn-danger">Stop Demo</button>
        </div>
        <div class="say-row">
          <input type="text" id="say-input" placeholder="Type something for Reachy to say...">
          <button id="btn-say">Say</button>
        </div>
      </div>
    </div>
    </div><!-- /tab-robot -->
    <div class="tabpane" id="tab-people">
    <div class="card">
      <h2>People Reachy Remembers</h2>
      <div id="people-list">&mdash;</div>
    </div>
    </div><!-- /tab-people -->
    <div class="tabpane" id="tab-admin">
    <!-- Business-sensitive info (provider, model, spend) lives on its own tab
         so the default view is safe to project to visitors/kids. -->
    <div class="card">
      <h2>LLM &amp; Costs</h2>
      <div class="status-row">
        <span class="status-label">LLM Provider</span>
        <span class="status-value"><span class="chip chip-groq" id="llm-provider">groq</span></span>
      </div>
      <div class="status-row">
        <span class="status-label">Model</span>
        <span class="status-value mono" id="llm-model">&mdash;</span>
      </div>
      <div class="status-row">
        <span class="status-label">Tokens In</span>
        <span class="status-value mono" id="tokens-in">0</span>
      </div>
      <div class="status-row">
        <span class="status-label">Tokens Out</span>
        <span class="status-value mono" id="tokens-out">0</span>
      </div>
      <div class="status-row">
        <span class="status-label">Est. Cost</span>
        <span class="status-value mono" id="est-cost">$0.0000</span>
      </div>
    </div>
    </div><!-- /tab-admin -->
  </div>
</main>
<script>
var TABS = ["robot", "people", "admin"];
function showTab(name) {
  TABS.forEach(function(t) {
    document.getElementById("tab-" + t).className = "tabpane" + (t === name ? " active" : "");
    document.getElementById("tab-btn-" + t).className = "tab-btn" + (t === name ? " active" : "");
  });
  if (name === "people") { refreshPeople(); }
}

function refreshPeople() {
  fetch("/api/people")
    .then(function(r) { return r.json(); })
    .then(function(data) {
      var el = document.getElementById("people-list");
      if (!data.people || !data.people.length) {
        el.textContent = "Nobody remembered yet — say hi to Reachy!";
        return;
      }
      el.innerHTML = "";
      data.people.forEach(function(p) {
        var block = document.createElement("div");
        block.className = "person-block";
        var h = document.createElement("h3");
        h.textContent = p.name;
        block.appendChild(h);
        var ul = document.createElement("ul");
        (p.facts.length ? p.facts : ["(no facts yet)"]).forEach(function(f) {
          var li = document.createElement("li");
          li.textContent = f;
          ul.appendChild(li);
        });
        block.appendChild(ul);
        el.appendChild(block);
      });
    })
    .catch(function() {});
}
setInterval(function() {
  if (document.getElementById("tab-people").className.indexOf("active") !== -1) {
    refreshPeople();
  }
}, 5000);

var STATE_BANNER = {
  idle:      "😴 Reachy is sleeping",
  listening: "🎤 You can talk now!",
  thinking:  "🤔 Reachy is thinking…",
  speaking:  "🗣️ Reachy is talking…"
};
var muted = false;
var connOK = true;
var pollTimer = null;
var pollDelay = 500;
var failCount = 0;
var ws = null;
var wsActive = false;
var wsRetryTimer = null;
var vid = document.getElementById("video-feed");
var vidOK = true;
var vidRetryTimer = null;
var ANIM_STATES = ["idle", "listening", "thinking", "speaking"];
document.body.dataset.state = "idle";

function post(path, body) {
  return fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: body ? JSON.stringify(body) : undefined
  });
}

function setConnBanner(show) {
  document.getElementById("conn-banner").classList.toggle("show", show);
}
var dotTimer = null;
function startDots(el) {
  if (dotTimer) return;
  var n = 0;
  el.textContent = "";
  dotTimer = setInterval(function() { n = (n+1)%4; el.textContent = ".".repeat(n); }, 400);
}
function stopDots() {
  if (dotTimer) { clearInterval(dotTimer); dotTimer = null; }
}

function videoError() {
  vidOK = false;
  document.getElementById("video-overlay").classList.add("show");
  startDots(document.getElementById("vid-dots"));
  if (vidRetryTimer) clearTimeout(vidRetryTimer);
  vidRetryTimer = setTimeout(reconnectVideo, 2000);
}
function videoOK() {
  vidOK = true;
  document.getElementById("video-overlay").classList.remove("show");
  stopDots();
  if (vidRetryTimer) { clearTimeout(vidRetryTimer); vidRetryTimer = null; }
}
function reconnectVideo() {
  vid.src = "/video?ts=" + Date.now();
  vidRetryTimer = setTimeout(function() {
    if (!vidOK) videoError();
  }, 5000);
}

document.getElementById("btn-wake").onclick = function() { post("/api/wake"); };
document.getElementById("btn-sleep").onclick = function() { post("/api/sleep"); };
document.getElementById("btn-stop").onclick = function() {
  if (confirm("Stop the whole demo? Reachy will go to sleep and the program will exit.")) {
    post("/api/stop");
  }
};
document.getElementById("btn-mute").onclick = function() {
  muted = !muted;
  post("/api/mute", {muted: muted});
  document.getElementById("btn-mute").textContent = muted ? "Unmute" : "Mute";
};
document.getElementById("btn-say").onclick = function() {
  var input = document.getElementById("say-input");
  var text = input.value.trim();
  if (!text) return;
  post("/api/say", {text: text});
  input.value = "";
};
document.getElementById("say-input").addEventListener("keydown", function(e) {
  if (e.key === "Enter") document.getElementById("btn-say").click();
});

function connectionUp() {
  failCount = 0;
  pollDelay = 500;
  if (!connOK) {
    connOK = true;
    setConnBanner(false);
    stopDots();
    if (!vidOK) reconnectVideo();
  }
}

function render(s) {
      var dot = document.getElementById("robot-dot");
      dot.className = "dot " + (s.robot_online ? "dot-on" : "dot-off");
      document.getElementById("robot-status").textContent = s.robot_online ? "Online" : "Offline";

      var animState = ANIM_STATES.indexOf(s.anim_state) !== -1 ? s.anim_state : "idle";
      if (!s.robot_online) { animState = "idle"; }  // asleep: never show "you can talk"
      document.body.dataset.state = animState;
      document.getElementById("state-banner").textContent =
        STATE_BANNER[animState] || STATE_BANNER.idle;

      var badge = document.getElementById("anim-badge");
      badge.textContent = s.anim_state;
      badge.className = "badge badge-" + animState;

      document.getElementById("current-lang").textContent = s.current_lang;
      document.getElementById("last-user").textContent = s.last_user || "\u2014";
      document.getElementById("last-reply").textContent = s.last_reply || "\u2014";

      document.getElementById("last-face-name").textContent = s.last_face_name;
      var conf = document.getElementById("last-face-conf");
      conf.textContent = s.last_face_conf > 0 ? " (" + (s.last_face_conf * 100).toFixed(0) + "%)" : "";

      document.getElementById("turn-count").textContent = s.turn_count;
      var m = Math.floor(s.uptime_s / 60);
      var sec = Math.floor(s.uptime_s % 60);
      document.getElementById("uptime").textContent =
        String(m).padStart(2, "0") + ":" + String(sec).padStart(2, "0");

      document.getElementById("lat-stt").textContent = s.stt_s.toFixed(2) + "s";
      document.getElementById("lat-llm").textContent = s.llm_ttf_s.toFixed(2) + "s";
      document.getElementById("lat-tts").textContent = s.tts_tta_s.toFixed(2) + "s";
      document.getElementById("lat-total").textContent = s.total_s.toFixed(2) + "s";

      document.getElementById("faces-visible").textContent = s.faces_visible;
      document.getElementById("known-person-count").textContent = s.known_person_count;

      var prov = document.getElementById("llm-provider");
      prov.textContent = s.llm_provider;
      prov.className = "chip chip-" + s.llm_provider;

      muted = s.muted;
      document.getElementById("mic-status").textContent = muted ? "Muted" : "Unmuted";
      document.getElementById("btn-mute").textContent = muted ? "Unmute" : "Mute";

      document.getElementById("rec").classList.toggle("active", s.anim_state === "speaking");

      document.getElementById("llm-model").textContent = s.llm_model || "—";
      document.getElementById("tokens-in").textContent = s.tokens_in;
      document.getElementById("tokens-out").textContent = s.tokens_out;
      document.getElementById("est-cost").textContent = "$" + s.est_cost_usd.toFixed(4);
      document.getElementById("person-summary").textContent = s.person_summary || "—";
}

/* --- /status polling: fallback transport when the WebSocket is down --- */
function startPolling() {
  if (!pollTimer) pollTimer = setTimeout(poll, 100);
}
function stopPolling() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
}
function poll() {
  if (wsActive) { pollTimer = null; return; }
  fetch("/status")
    .then(function(r) { return r.json(); })
    .then(function(s) {
      connectionUp();
      render(s);
    })
    .catch(function() {
      failCount++;
      if (failCount >= 2 && connOK) {
        connOK = false;
        setConnBanner(true);
        startDots(document.getElementById("conn-dots"));
      }
      pollDelay = Math.min(failCount * 500, 3000);
    })
    .finally(function() {
      pollTimer = wsActive ? null : setTimeout(poll, pollDelay);
    });
}

/* --- WebSocket: primary transport; server pushes snapshots on change --- */
function connectWS() {
  if (ws) return;
  if (wsRetryTimer) { clearTimeout(wsRetryTimer); wsRetryTimer = null; }
  var sock;
  try {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    sock = new WebSocket(proto + "//" + location.host + "/ws");
  } catch (e) {
    wsDown();
    return;
  }
  ws = sock;
  sock.onopen = function() {
    wsActive = true;
    connectionUp();
    stopPolling();
  };
  sock.onmessage = function(ev) {
    var s;
    try { s = JSON.parse(ev.data); } catch (e) { return; }
    connectionUp();
    render(s);
  };
  sock.onerror = function() {
    try { sock.close(); } catch (e) {}
  };
  sock.onclose = function() {
    if (ws === sock) ws = null;
    wsActive = false;
    wsDown();
  };
}
function wsDown() {
  startPolling();
  if (!wsRetryTimer) {
    wsRetryTimer = setTimeout(function() {
      wsRetryTimer = null;
      connectWS();
    }, 5000);
  }
}

function visibilityHack() {
  if (document.visibilityState === "visible") {
    if (!vidOK) reconnectVideo();
    if (!ws && !wsActive) connectWS();
    if (!wsActive) startPolling();
  }
}
document.addEventListener("visibilitychange", visibilityHack);

connectWS();
</script>
</body>
</html>"""


class WebDashboard:
    def __init__(
        self,
        state: LiveState,
        camera_hub: CameraHub,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        self.state = state
        self.camera_hub = camera_hub
        self.host = host
        self.port = port
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self.app = FastAPI(title="Reachy Mini Dashboard")
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.get("/")
        def _index() -> HTMLResponse:
            return HTMLResponse(_HTML.replace("__PORT__", str(self.port)))

        @self.app.get("/video")
        def _video() -> StreamingResponse:
            return StreamingResponse(
                self._video_stream(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.get("/status")
        def _status() -> dict:
            return self.state.snapshot()

        @self.app.websocket("/ws")
        async def _ws(websocket: WebSocket) -> None:
            # Push the state snapshot as JSON text frames ~8-10x/s, but only
            # when the serialized snapshot changed since the last push. Each
            # client gets its own coroutine, so simultaneous clients are fine.
            await websocket.accept()
            last_key: str | None = None
            last_sent = 0.0
            try:
                while not self._stop_flag.is_set():
                    snap = self.state.snapshot()
                    # Compare WITHOUT uptime_s — it ticks every iteration and
                    # made "change-only" push a no-op (full-rate to every
                    # client). A 1 s heartbeat still keeps uptime fresh.
                    key = json.dumps({k: v for k, v in snap.items()
                                      if k != "uptime_s"})
                    now = time.time()
                    if key != last_key or now - last_sent >= 1.0:
                        await websocket.send_text(json.dumps(snap))
                        last_key = key
                        last_sent = now
                    await asyncio.sleep(0.12)
            except WebSocketDisconnect:
                pass  # client closed the tab / navigated away — normal
            except Exception:
                pass  # connection reset mid-send etc. — not worth a traceback

        @self.app.post("/api/wake")
        def _wake() -> dict:
            self.state.request_wake()
            return {"ok": True}

        @self.app.post("/api/sleep")
        def _sleep() -> dict:
            self.state.request_sleep()
            return {"ok": True}

        @self.app.post("/api/say")
        async def _say(request: Request) -> dict:
            data = await request.json()
            text = data.get("text", "")
            self.state.request_say(text)
            return {"ok": True}

        @self.app.post("/api/stop")
        def _stop() -> dict:
            self.state.request_shutdown()
            return {"ok": True}

        @self.app.get("/api/people")
        def _people() -> dict:
            from reachy_demo.memory import known_people, load_person_facts
            return {"people": [
                {"name": n, "facts": load_person_facts(n)}
                for n in known_people()
            ]}

        @self.app.post("/api/mute")
        async def _mute(request: Request) -> dict:
            data = await request.json()
            self.state.muted = bool(data.get("muted", False))
            return {"ok": True}

    def _video_stream(self):
        heartbeat = 0
        while not self._stop_flag.is_set():
            try:
                jpg = self.camera_hub.mjpeg_bytes()
            except Exception:
                jpg = None
            if jpg is not None:
                heartbeat = 0
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n"
                    b"\r\n" + jpg + b"\r\n"
                )
            else:
                heartbeat += 1
                if heartbeat >= 60:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: 0\r\n"
                        b"\r\n\r\n"
                    )
                    heartbeat = 0
            time.sleep(1 / 30)

    def start(self) -> None:
        import uvicorn

        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=uvicorn.run,
            args=(self.app,),
            kwargs={
                "host": self.host,
                "port": self.port,
                "log_level": "warning",
            },
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
