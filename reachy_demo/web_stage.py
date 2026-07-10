"""reachy_demo/web_stage.py — tabbed web dashboard, split-view stage for projector."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.responses import FileResponse, HTMLResponse, StreamingResponse

if TYPE_CHECKING:
    from reachy_demo.camera import CameraHub
    from reachy_demo.face_id import FaceIdentifier
    from reachy_demo.live_state import LiveState

_DEFAULT_FACES_DIR = Path(__file__).parent.parent / "faces"


def _face_slug(name: str) -> str:
    """Match FaceIdentifier's on-disk folder naming, then strip anything unsafe
    for a single path segment. The resolve()+prefix check in /api/face is the
    real traversal guard; this is defense in depth."""
    slug = name.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_\-]", "", slug)

GESTURES = [
    "acknowledge", "yes", "no", "thank", "thinking", "curious", "confused",
    "greeting", "celebrate", "proud", "amazed", "love", "laugh", "oops",
    "shy", "surprised", "cheerful", "success", "relief",
]

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Reachy</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;background:#0a0a0a;color:#fff;font-family:system-ui,sans-serif;overflow:hidden}

#tabs{position:fixed;top:0;left:0;right:0;z-index:100;display:flex;background:#0f172a;border-bottom:2px solid #1e293b}
#tabs button{flex:1;max-width:200px;padding:10px 20px;background:transparent;border:none;color:#64748b;font-size:.9rem;font-weight:700;cursor:pointer;letter-spacing:1px;border-bottom:3px solid transparent;transition:.15s}
#tabs button:hover{color:#94a3b8}
#tabs button.active{color:#fff;border-bottom-color:#3b82f6;background:#111827}
.view{display:none;height:100vh;padding-top:48px}
.view.active{display:block}

/* ═══ STAGE (projector) ═══ */
#stage-view.active{display:grid;grid-template-columns:3fr 2fr;gap:8px;padding:56px 8px 8px;transition:background .8s}
#stage-view.speaking{background:linear-gradient(135deg,#0c1445,#0a1a2e,#0c1445)}
#stage-view.listening{background:linear-gradient(135deg,#0a2e0a,#0a1f0e,#0a2e0a)}
#stage-view.thinking{background:linear-gradient(135deg,#1f1408,#2a1a06,#1f1408)}
#stage-view.dancing{background:linear-gradient(135deg,#1a0a2e,#2e0a2e,#1a0a2e)}
#stage-view.idle{background:linear-gradient(135deg,#0a0a0a,#111,#0a0a0a)}

#cam-box{position:relative;background:#000;border-radius:12px;overflow:hidden;border:2px solid rgba(255,255,255,.1)}
#cam-box img{width:100%;height:100%;object-fit:cover}
#state-tag{position:absolute;top:10px;left:10px;padding:6px 16px;border-radius:999px;font-size:1.1rem;font-weight:900;letter-spacing:2px;backdrop-filter:blur(8px);background:rgba(0,0,0,.6);transition:background .3s,box-shadow .3s}
#state-tag.speaking{background:rgba(59,130,246,.5);box-shadow:0 0 20px rgba(59,130,246,.4)}
#state-tag.listening{background:rgba(34,197,94,.5);box-shadow:0 0 20px rgba(34,197,94,.4)}
#state-tag.thinking{background:rgba(245,158,11,.5);box-shadow:0 0 20px rgba(245,158,11,.4)}
#state-tag.dancing{background:rgba(168,85,247,.5);box-shadow:0 0 20px rgba(168,85,247,.4)}
#face-bar{position:absolute;bottom:0;left:0;right:0;display:flex;gap:6px;padding:12px;background:rgba(0,0,0,.7);overflow-x:auto;min-height:64px;align-items:center}
.face-tag{display:flex;align-items:center;gap:8px;padding:8px 20px;border-radius:999px;font-size:2rem;font-weight:900;white-space:nowrap;flex-shrink:0;animation:popin .3s ease}
@keyframes popin{from{transform:scale(.5);opacity:0}to{transform:scale(1);opacity:1}}
.face-tag.known{background:rgba(34,197,94,.3);border:2px solid rgba(34,197,94,.6);color:#bbf7d0}
.face-tag.visitor{background:rgba(96,165,250,.3);border:2px solid rgba(96,165,250,.6);color:#bfdbfe}

/* ── right info panel ── */
#info{display:flex;flex-direction:column;gap:8px;overflow:hidden}
.card{background:rgba(255,255,255,.04);border-radius:14px;padding:16px 18px;border:1px solid rgba(255,255,255,.08)}

#big-status{display:flex;align-items:center;gap:14px;min-height:90px}
#big-status .icon{font-size:3.2rem;animation:float 2s ease-in-out infinite}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
#big-status .txt{font-size:1.5rem;font-weight:800;line-height:1.3}
#big-status .txt .sub{font-size:.85rem;opacity:.5;font-weight:400;display:block;margin-top:4px}

#caption{font-size:1.6rem;font-weight:700;line-height:1.3;min-height:60px;display:flex;align-items:center;transition:all .3s}
#caption .blink{display:inline-block;width:3px;height:1.6rem;background:currentColor;margin-left:4px;animation:blk .8s steps(2) infinite}
@keyframes blk{50%{opacity:0}}
#caption.speaking{color:#bfdbfe}#caption.thinking{color:#fbbf24;font-style:italic}#caption.listening{color:rgba(255,255,255,.3);font-size:1.1rem}#caption.empty{color:rgba(255,255,255,.15);font-size:.9rem}

/* ── hear→understand→think→speak process (stage, kid-facing) ── */
#pipeline{display:flex;flex-direction:column;gap:6px;padding:12px 14px}
.prow{display:flex;gap:11px;align-items:flex-start;opacity:.28;border-radius:10px;padding:6px 9px;transition:opacity .3s,background .3s}
.prow.on{opacity:1}
.prow.on .pr-ic{animation:pop .35s ease}
@keyframes pop{0%{transform:scale(.7)}60%{transform:scale(1.15)}100%{transform:scale(1)}}
#pr-hear.on{background:rgba(34,197,94,.14)}
#pr-words.on{background:rgba(16,185,129,.14)}
#pr-think.on{background:rgba(245,158,11,.14)}
#pr-say.on{background:rgba(59,130,246,.16)}
.pr-ic{font-size:1.6rem;line-height:1.2;flex-shrink:0}
.pr-body{flex:1;min-width:0}
.pr-lbl{font-size:.58rem;text-transform:uppercase;letter-spacing:.07em;opacity:.55;font-weight:700;margin-bottom:1px}
.pr-txt{font-size:1.05rem;font-weight:700;line-height:1.22;word-wrap:break-word;overflow-wrap:anywhere}
#pr-words-t{color:#a7f3d0}#pr-think-t{font-style:italic;color:#fbbf24}#pr-say-t{color:#bfdbfe}
.pr-txt .blink{display:inline-block;width:3px;height:1.05rem;background:currentColor;margin-left:3px;animation:blk .8s steps(2) infinite;vertical-align:middle}

#fact-bar{display:flex;align-items:center;gap:8px;font-size:.9rem;color:#94a3b8;min-height:30px}
#fact-bar .icon{font-size:1.2rem}

#face-list{display:flex;gap:6px;flex-wrap:wrap;min-height:28px;align-items:center}
.f-badge{display:flex;align-items:center;gap:4px;padding:4px 10px;border-radius:999px;font-size:.85rem;font-weight:700;animation:popin .3s ease}
.f-badge.know{background:rgba(34,197,94,.2);border:1px solid rgba(34,197,94,.5);color:#bbf7d0}
.f-badge.new{background:rgba(96,165,250,.2);border:1px solid rgba(96,165,250,.5);color:#bfdbfe}

#stage-foot{display:flex;justify-content:space-between;font-size:.65rem;color:#374151;padding:0 4px}

/* ═══ CONTROL (laptop) ═══ */
#control-view.active{display:block;overflow:auto;padding:60px 14px 14px}
.ctrl-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;max-width:1400px}
.cd{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px}
.cd h2{font-size:.72rem;text-transform:uppercase;letter-spacing:1.5px;color:#94a3b8;margin-bottom:10px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px}
b{border-radius:6px;padding:7px 12px;cursor:pointer;font-size:.82rem;font-weight:600;border:1px solid;transition:.12s}
b.df{background:#334155;color:#e5e7eb;border-color:#475569}
b.df:hover{background:#475569;border-color:#64748b}
b.out-btn{font-size:.78rem;padding:3px 9px;opacity:.5}
b.out-btn.on{background:#1e3a8a;border-color:#3b82f6;color:#bfdbfe;opacity:1}
b.go{background:#14532d;border-color:#166534;color:#bbf7d0}
b.go:hover{background:#166534}
b.warn{background:#78350f;border-color:#92400e;color:#fde68a}
b.warn:hover{background:#92400e}
b.danger{background:#7f1d1d;border-color:#991b1b;color:#fecaca}
b.danger:hover{background:#991b1b}
input[type=text]{background:#0f172a;border:1px solid #475569;color:#fff;border-radius:6px;padding:8px 10px;font-size:.9rem;flex:1;min-width:120px}
input[type=text]:focus{outline:none;border-color:#60a5fa}
.ggrid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
.ggrid b{padding:6px 4px;font-size:.75rem;text-align:center}
.stat{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #334155;font-size:.82rem}
.stat:last-child{border-bottom:none}
.stat .k{color:#94a3b8}.stat .v{color:#fff;font-weight:600;font-variant-numeric:tabular-nums}
.toggle{display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none}
.toggle .sw{width:38px;height:22px;background:#334155;border-radius:999px;position:relative;transition:.2s}
.toggle .sw::after{content:'';position:absolute;top:2px;left:2px;width:18px;height:18px;background:#fff;border-radius:50%;transition:.2s}
.toggle.on .sw{background:#22c55e}.toggle.on .sw::after{left:18px}
.people{max-height:320px;overflow:auto}
.pgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:10px}
.pcard{background:#0f172a;border:1px solid #334155;border-radius:8px;padding:8px;text-align:center}
.pcard img,.pcard .noimg{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;background:#1e293b;display:flex;align-items:center;justify-content:center;font-size:1.8rem;margin-bottom:6px}
.pcard .n{font-weight:700;color:#fff;font-size:.8rem;word-break:break-word;margin-bottom:4px}
.pcard .f{font-size:.68rem;color:#94a3b8;margin-bottom:6px}
.pcard .actions{display:flex;gap:4px;justify-content:center}
.pcard .actions button{flex:1;border:1px solid #475569;background:#1e293b;color:#e5e7eb;border-radius:5px;padding:4px 2px;font-size:.85rem;cursor:pointer}
.pcard .actions button:hover{background:#334155}
.cam-prev{width:100%;border-radius:8px;border:1px solid #334155;margin-bottom:8px}

/* ═══ TECH (sound-check audit) ═══ */
#tech-view.active{display:block;overflow:auto;padding:60px 14px 14px}
#scope-cv{width:100%;height:220px;background:#000;border-radius:8px;display:block}
#scope-status{font-size:.72rem;color:#94a3b8;margin-top:6px}
#pipe-strip{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.pill{padding:8px 14px;border-radius:10px;background:#334155;border:1px solid #475569;font-size:.78rem;font-weight:700;color:#94a3b8;min-width:90px;text-align:center;transition:.2s}
.pill.on{background:rgba(34,197,94,.25);border-color:rgba(34,197,94,.6);color:#bbf7d0}
.pill.off{background:rgba(239,68,68,.25);border-color:rgba(239,68,68,.6);color:#fecaca}
.pill .sub{display:block;font-size:.65rem;font-weight:400;opacity:.85;margin-top:2px}
.chev{color:#475569;font-size:1.1rem}
.chip{padding:2px 8px;border-radius:999px;font-size:.68rem;font-weight:800;margin-left:6px}
.chip.pass{background:rgba(34,197,94,.2);color:#bbf7d0;border:1px solid rgba(34,197,94,.5)}
.chip.fail{background:rgba(239,68,68,.2);color:#fecaca;border:1px solid rgba(239,68,68,.5)}
</style></head><body>
<div id="tabs">
  <button id="tab-stage" onclick="showTab('stage')">Stage</button>
  <button id="tab-control" onclick="showTab('control')">Control</button>
  <button id="tab-tech" onclick="showTab('tech')">Tech</button>
</div>
<div id="stage-view" class="view">
  <div id="cam-box">
    <img id="feed-s" src="/video" onerror="r('feed-s')">
    <div id="state-tag">IDLE</div>
    <div id="face-bar"><span style="color:rgba(255,255,255,.2)">waiting for friends</span></div>
  </div>
  <div id="info">
    <div id="big-status" class="card"><div class="icon" id="big-icon">🤖</div><div class="txt" id="big-txt">Ready!<span class="sub">Reachy is waiting to meet you</span></div></div>
    <div id="pipeline" class="card">
      <div class="prow" id="pr-hear"><span class="pr-ic">👂</span><div class="pr-body"><div class="pr-lbl">1 · Hearing you</div><div class="pr-txt" id="pr-hear-t">…</div></div></div>
      <div class="prow" id="pr-words"><span class="pr-ic">📝</span><div class="pr-body"><div class="pr-lbl">2 · Understanding the words</div><div class="pr-txt" id="pr-words-t">…</div></div></div>
      <div class="prow" id="pr-think"><span class="pr-ic">🧠</span><div class="pr-body"><div class="pr-lbl">3 · Thinking of an answer</div><div class="pr-txt" id="pr-think-t">…</div></div></div>
      <div class="prow" id="pr-say"><span class="pr-ic">💬</span><div class="pr-body"><div class="pr-lbl">4 · Talking back</div><div class="pr-txt" id="pr-say-t">…</div></div></div>
    </div>
    <div id="fact-bar" class="card"><span class="icon">💡</span><span id="fact-txt">Reachy wants arms and legs someday!</span></div>
    <div id="face-list" class="card"><span style="color:rgba(255,255,255,.2)">come say hello!</span></div>
  </div>
  <div id="stage-foot"><span>Network School</span><span>Reachy Mini · <span id="s-lang">-</span></span></div>
</div>
<div id="control-view" class="view">
  <h1 style="font-size:1.1rem;color:#fff;margin-bottom:12px">Reachy Control Panel</h1>
  <div class="ctrl-grid">
    <div class="cd">
      <h2>Commands</h2>
      <div class="row">
        <b class="go" onclick="P('/api/wake')">Wake</b>
        <b class="warn" onclick="P('/api/sleep')">Sleep</b>
        <b class="danger" onclick="P('/api/stop')">Stop</b>
      </div>
      <div class="row">
        <label class="toggle" id="kid-t"><div class="sw"></div><span>Kid mode</span></label>
        <label class="toggle" id="mute-t"><div class="sw"></div><span>Mute</span></label>
        <label class="toggle" id="crowd-t"><div class="sw"></div><span>🧑‍🤝‍🧑 Crowd mode</span></label>
      </div>
      <div style="font-size:.68rem;color:#94a3b8;margin:-2px 0 2px">Crowd mode = ignore faces (no head-tracking/greeting), just talk. Flip it when the room fills up.</div>
      <h2 style="margin-top:14px">Make Reachy say</h2>
      <div class="row"><input type="text" id="say-i" placeholder="Type a line, Enter to fire" onkeydown="if(event.key==='Enter')S()"><b class="go" onclick="S()">Say</b></div>
      <h2 style="margin-top:14px">Quick phrases</h2>
      <div class="row" id="ph-grid"></div>
    </div>
    <div class="cd">
      <h2>Gestures (19)</h2>
      <div class="ggrid" id="g-grid"></div>
      <h2 style="margin-top:14px">Dances</h2>
      <div class="row" id="d-grid"></div>
    </div>
    <div class="cd">
      <h2>Camera</h2>
      <img class="cam-prev" src="/video" onerror="this.src='/video?t='+Date.now()">
      <h2>Status</h2>
      <div class="stat"><span class="k">Robot</span><span class="v" id="c-rob">-</span></div>
      <div class="stat"><span class="k">State</span><span class="v" id="c-st">-</span></div>
      <div class="stat"><span class="k">Faces</span><span class="v" id="c-fc">0</span></div>
      <div class="stat"><span class="k">Known</span><span class="v" id="c-kn">0</span></div>
      <div class="stat"><span class="k">Lang</span><span class="v" id="c-lg">-</span></div>
      <div class="stat"><span class="k">Turn</span><span class="v" id="c-tn">0</span></div>
      <div class="stat"><span class="k">Uptime</span><span class="v" id="c-up">0s</span></div>
    </div>
    <div class="cd">
      <h2>Latency</h2>
      <div class="stat"><span class="k">STT</span><span class="v" id="c-stt">-</span></div>
      <div class="stat"><span class="k">LLM TTF</span><span class="v" id="c-ttf">-</span></div>
      <div class="stat"><span class="k">TTS TTA</span><span class="v" id="c-tta">-</span></div>
      <div class="stat"><span class="k">Provider</span><span class="v" id="c-prv">-</span></div>
      <div class="stat"><span class="k">Model</span><span class="v" id="c-mod">-</span></div>
    </div>
    <div class="cd">
      <h2>Tokens &amp; cost</h2>
      <div class="stat"><span class="k">In</span><span class="v" id="c-ti">0</span></div>
      <div class="stat"><span class="k">Out</span><span class="v" id="c-to">0</span></div>
      <div class="stat"><span class="k">Cost</span><span class="v" id="c-co">$0</span></div>
      <div class="stat"><span class="k">Last user</span><span class="v" id="c-lu" style="font-size:.7rem;max-width:60%">-</span></div>
      <div class="stat"><span class="k">Last reply</span><span class="v" id="c-lr" style="font-size:.7rem;max-width:60%">-</span></div>
    </div>
    <div class="cd">
      <h2>Audio &amp; energy</h2>
      <div class="stat"><span class="k">Speaker</span><span class="v">
        <b class="df out-btn on" id="out-robot" onclick="setOut('robot')">🤖 Robot</b>
        <b class="df out-btn" id="out-proj" onclick="setOut('projector')">📽️ Projector</b>
      </span></div>
      <div style="font-size:.68rem;color:#94a3b8;margin:2px 0 8px">Projector = HDMI speaker, much louder for a big room.</div>
      <div class="stat"><span class="k">Volume</span><span class="v" id="s-vol">2.5</span></div>
      <input type="range" id="vol-sl" min="0" max="5" step=".1" value="2.5" style="width:100%;margin-bottom:8px" oninput="dV('vol',this.value)">
      <div class="stat"><span class="k">Rate</span><span class="v" id="s-rate">+20%</span></div>
      <input type="range" id="rate-sl" min="-30" max="50" step="5" value="20" style="width:100%;margin-bottom:8px" oninput="dV('rate',this.value)">
      <div class="stat"><span class="k">Energy</span><span class="v" id="s-ene">1.0</span></div>
      <input type="range" id="ene-sl" min="0" max="1" step=".1" value="1" style="width:100%">
    </div>
    <div class="cd" style="grid-column:1/-1">
      <h2>Enrolled people</h2>
      <div class="people" id="ppl">loading...</div>
    </div>
  </div>
</div>
<div id="tech-view" class="view">
  <h1 style="font-size:1.1rem;color:#fff;margin-bottom:12px">Audio Pipeline Tech Audit</h1>
  <div class="cd" style="margin-bottom:12px">
    <h2>Oscilloscope</h2>
    <canvas id="scope-cv"></canvas>
    <div id="scope-status">connecting...</div>
  </div>
  <div class="cd" style="margin-bottom:12px">
    <h2>Pipeline</h2>
    <div id="pipe-strip">
      <div class="pill on" id="p-mic"><div>MIC</div><span class="sub" id="p-mic-v">-</span></div>
      <span class="chev">&#8250;</span>
      <div class="pill" id="p-vad"><div>VAD</div><span class="sub" id="p-vad-v">-</span></div>
      <span class="chev">&#8250;</span>
      <div class="pill" id="p-gate"><div>GATE</div><span class="sub" id="p-gate-v">-</span></div>
      <span class="chev">&#8250;</span>
      <div class="pill" id="p-stt"><div>STT</div><span class="sub" id="p-stt-v">-</span></div>
      <span class="chev">&#8250;</span>
      <div class="pill" id="p-llm"><div>LLM</div><span class="sub" id="p-llm-v">-</span></div>
      <span class="chev">&#8250;</span>
      <div class="pill" id="p-tts"><div>TTS</div><span class="sub" id="p-tts-v">-</span></div>
    </div>
  </div>
  <div class="ctrl-grid">
    <div class="cd">
      <h2>Mic tuning (sound-check)</h2>
      <div style="font-size:.72rem;color:#94a3b8;margin-bottom:8px">Higher = ignore background chatter (do a 30s sound-check)</div>
      <div class="stat"><span class="k">Noise floor</span><span class="v" id="s-rms">120</span></div>
      <input type="range" id="rms-sl" min="0" max="2000" step="10" value="120" style="width:100%;margin-bottom:8px" oninput="dA('rms',this.value)">
      <div class="stat"><span class="k">Voiced ratio</span><span class="v" id="s-voi">.30</span></div>
      <input type="range" id="voi-sl" min="0" max="1" step=".01" value=".30" style="width:100%;margin-bottom:8px" oninput="dA('voi',this.value)">
      <div class="stat"><span class="k">Voice peak</span><span class="v" id="s-pk">.75</span></div>
      <input type="range" id="pk-sl" min="0" max="1" step=".01" value=".75" style="width:100%;margin-bottom:8px" oninput="dA('pk',this.value)">
      <div class="stat"><span class="k">Min duration</span><span class="v" id="s-dur">.30</span></div>
      <input type="range" id="dur-sl" min="0" max="1.5" step=".05" value=".30" style="width:100%;margin-bottom:8px" oninput="dA('dur',this.value)">
      <div class="stat"><span class="k">Mic trigger</span><span class="v" id="s-vad">.45</span></div>
      <input type="range" id="vad-sl" min="0.1" max="0.95" step=".01" value=".45" style="width:100%;margin-bottom:8px" oninput="dA('vad',this.value)">
      <div class="stat"><span class="k">Barge-in</span><span class="v" id="s-brg">.75</span></div>
      <input type="range" id="brg-sl" min="0.1" max="0.95" step=".01" value=".75" style="width:100%" oninput="dA('brg',this.value)">
    </div>
    <div class="cd">
      <h2>Last gate decision</h2>
      <div id="gate-headline" style="font-weight:800;font-size:1rem;margin-bottom:8px">-</div>
      <div class="stat"><span class="k">Energy / RMS</span><span class="v"><span id="g-rms">-</span><span class="chip" id="g-rms-c">-</span></span></div>
      <div class="stat"><span class="k">Voiced ratio</span><span class="v"><span id="g-voi">-</span><span class="chip" id="g-voi-c">-</span></span></div>
      <div class="stat"><span class="k">Voice peak</span><span class="v"><span id="g-pk">-</span><span class="chip" id="g-pk-c">-</span></span></div>
      <div class="stat"><span class="k">Duration</span><span class="v"><span id="g-dur">-</span><span class="chip" id="g-dur-c">-</span></span></div>
    </div>
    <div class="cd">
      <h2>Last turn</h2>
      <div class="stat"><span class="k">Language</span><span class="v" id="t-lang">-</span></div>
      <div class="stat"><span class="k">Last heard</span><span class="v" id="t-lu" style="font-size:.7rem;max-width:60%">-</span></div>
      <div class="stat"><span class="k">Last reply</span><span class="v" id="t-lr" style="font-size:.7rem;max-width:60%">-</span></div>
      <div class="stat"><span class="k">STT</span><span class="v" id="t-stt">-</span></div>
      <div class="stat"><span class="k">LLM TTF</span><span class="v" id="t-ttf">-</span></div>
      <div class="stat"><span class="k">TTS TTA</span><span class="v" id="t-tta">-</span></div>
      <div class="stat"><span class="k">Reply wait</span><span class="v" id="t-wait">-</span></div>
      <div class="stat"><span class="k">Talk time</span><span class="v" id="t-talk">-</span></div>
    </div>
  </div>
</div>
<script>
"use strict";
const $=id=>document.getElementById(id);
const G=__GESTURES__;
const K={idle:'IDLE',listening:'Listening',thinking:'Thinking',speaking:'Speaking',dancing:'Dancing'};
const I={idle:'💤',listening:'👂',thinking:'🧠',speaking:'💬',dancing:'🕺'};
const M={idle:'Ready!',listening:'Listening!',thinking:'Thinking...',speaking:'Speaking!',dancing:'Dancing!'};
const F=['Reachy wants arms and legs someday!','Reachy lost brother Pixel vanished one firmware update ago!','Network School has 2,000+ members from 80+ countries!','Reachy can speak ANY language — try it!','Reachy antennas are like little hands waving hello!'];

const TABS=['stage','control','tech'];
function showTab(n){
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  $(n+'-view').classList.add('active');
  TABS.forEach(t=>$('tab-'+t).classList.toggle('active',t===n));
  localStorage.setItem('r-tab',n);
}
(function(){
  const h=location.hash.replace('#','');
  const s=localStorage.getItem('r-tab')||'control';
  showTab(TABS.includes(h)?h:s);
})();
window.addEventListener('hashchange',()=>{const h=location.hash.replace('#','');showTab(TABS.includes(h)?h:'control');});

function r(id){const el=$(id);setTimeout(()=>{el.src='/video?t='+Date.now();},1500);}

let ws;function W(){
  ws=new WebSocket('ws://'+location.host+'/ws');
  ws.onmessage=e=>{try{R(JSON.parse(e.data));}catch(err){console.error('render',err);}};
  ws.onclose=()=>setTimeout(W,1500);ws.onerror=()=>ws.close();
}W();

let scopeWs,scopeBuf=[],scopeConnected=false;
function connectScope(){
  scopeWs=new WebSocket('ws://'+location.host+'/scope');
  scopeWs.onopen=()=>{scopeConnected=true;$('scope-status').textContent='live';};
  scopeWs.onmessage=e=>{
    try{
      const d=JSON.parse(e.data);
      scopeBuf.push(d);
      if(scopeBuf.length>300)scopeBuf.shift();
    }catch(err){console.error('scope',err);}
  };
  scopeWs.onclose=()=>{scopeConnected=false;$('scope-status').textContent='reconnecting…';setTimeout(connectScope,1500);};
  scopeWs.onerror=()=>scopeWs.close();
}
connectScope();

function drawScope(){
  requestAnimationFrame(drawScope);
  const cv=$('scope-cv');
  if(!cv)return;
  if(cv.clientWidth&&cv.width!==cv.clientWidth){cv.width=cv.clientWidth;cv.height=220;}
  const ctx=cv.getContext('2d');
  const w=cv.width,h=cv.height;
  if(!w||!h)return;
  ctx.fillStyle='#000';ctx.fillRect(0,0,w,h);
  if(!scopeBuf.length){
    ctx.fillStyle='#475569';ctx.font='13px system-ui,sans-serif';
    ctx.fillText(scopeConnected?'waiting for signal…':'reconnecting…',10,h/2);
    return;
  }
  const n=scopeBuf.length;
  const N=300;
  const xw=w/N;
  let peak=0;for(let i=0;i<n;i++)peak=Math.max(peak,scopeBuf[i].rms||0);
  const ymax=Math.max(1500,peak*1.2);
  const floor=scopeBuf[n-1].floor||0;

  // in-speech shading + trigger markers
  let prev=false;
  for(let i=0;i<n;i++){
    const d=scopeBuf[i];
    const x=w-(n-i)*xw;
    if(d.in_speech){ctx.fillStyle='rgba(34,197,94,.15)';ctx.fillRect(x,0,xw+1,h);}
    if(d.in_speech&&!prev){
      ctx.strokeStyle='#4ade80';ctx.lineWidth=2;
      ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();
      ctx.fillStyle='#4ade80';ctx.font='bold 11px system-ui,sans-serif';
      ctx.fillText('▲ TRIGGER',Math.max(2,Math.min(x+4,w-80)),14);
    }
    prev=d.in_speech;
  }

  // noise-floor line
  const floorY=h-Math.min(1,floor/ymax)*h;
  ctx.strokeStyle='#ef4444';ctx.lineWidth=1;ctx.setLineDash([4,3]);
  ctx.beginPath();ctx.moveTo(0,floorY);ctx.lineTo(w,floorY);ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#ef4444';ctx.font='11px system-ui,sans-serif';
  ctx.fillText('noise floor '+Math.round(floor),6,Math.max(12,floorY-4));

  // rms trace
  ctx.strokeStyle='#60a5fa';ctx.lineWidth=2;ctx.beginPath();
  for(let i=0;i<n;i++){
    const d=scopeBuf[i];
    const x=w-(n-i)*xw;
    const y=h-Math.min(1,(d.rms||0)/ymax)*h;
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
  }
  ctx.stroke();

  // current RMS readout
  const last=scopeBuf[n-1];
  ctx.fillStyle='#bfdbfe';ctx.font='bold 13px system-ui,sans-serif';
  ctx.fillText('RMS '+Math.round(last.rms||0),Math.max(4,w-110),16);
}
requestAnimationFrame(drawScope);

let ls='',lsp='',lfn='',ft=0,fi=0,kid=true,mute=false,crowd=false,vd=false,rd=false,ed=false;
let rmsd=false,void_=false,pkd=false,durd=false,vadd=false,brgd=false;

function R(s){
  const st=s.anim_state||'idle';
  const sv=$('stage-view');
  sv.className='view'+(sv.classList.contains('active')?' active':'')+' '+st;
  $('state-tag').textContent=K[st]||st;
  $('state-tag').className=st;
  $('big-icon').textContent=I[st]||'🤖';
  $('big-txt').innerHTML=M[st]+'<span class="sub">'+
    {idle:'Reachy is ready to meet you',listening:'Reachy hears you — talk to me!',thinking:'Reachy is figuring it out...',speaking:'Reachy is replying now!',dancing:'Party time!'}[st]+'</span>';

  // 4-step process the kids can follow: Hear → Understand → Think → Talk.
  // STT and LLM both live in anim_state 'thinking'; we split them by whether the
  // LLM has started streaming (llm_partial): empty = still turning sound into words.
  const think=s.llm_partial||'';
  const sttPhase=(st==='thinking' && !think);   // got audio, working out the words
  const llmPhase=(st==='thinking' && !!think);  // words known, composing a reply
  // 1 · Hearing
  $('pr-hear-t').textContent = st==='listening' ? (s.vad_in_speech?'I hear you! 🎤':'Listening for a voice…')
                             : (st==='idle'?'…':'Got it! 👂');
  // 2 · Understanding the words (STT transcript)
  $('pr-words-t').innerHTML = s.last_user ? esc(s.last_user)
                            : (sttPhase?'turning sound into words…<span class="blink"></span>':'…');
  // 3 · Thinking (LLM stream)
  $('pr-think-t').innerHTML = think ? esc(think.slice(-140))+(llmPhase?'<span class="blink"></span>':'')
                            : (llmPhase?'…<span class="blink"></span>':'…');
  // 4 · Talking back (TTS). Blank while still thinking; keep the finished reply once idle.
  let say = st==='speaking' ? (s.current_speech||s.last_reply||'')
          : st==='thinking' ? '' : (s.last_reply||'');
  $('pr-say-t').innerHTML=(say?esc(say):'…')+(st==='speaking'?'<span class="blink"></span>':'');
  $('pr-hear').classList.toggle('on',st==='listening');
  $('pr-words').classList.toggle('on',sttPhase);
  $('pr-think').classList.toggle('on',llmPhase);
  $('pr-say').classList.toggle('on',st==='speaking'||st==='dancing');

  const n=Date.now();
  if(n-ft>7000){fi=(fi+1)%F.length;ft=n;$('fact-txt').textContent=F[fi];}

  const fc=s.faces_visible||0;
  const fn=s.last_face_name||'-';
  if(fc>0&&fn!=='-'&&s.last_face_conf>0.45){
    $('face-list').innerHTML='<div class="f-badge know">👋 '+esc(fn)+'</div>'+(fc>1?'<div class="f-badge new">+'+String(fc-1)+' more</div>':'');
  }else if(fc>0){
    $('face-list').innerHTML='<div class="f-badge new">👀 '+fc+' person(s) nearby</div>';
  }else{$('face-list').innerHTML='<span style="color:rgba(255,255,255,.2)">come say hello!</span>';}

  if(fc>0&&fn!=='-'&&s.last_face_conf>0.45&&fn!==lfn){
    $('face-bar').innerHTML='<div class="face-tag known">👋 '+esc(fn)+'</div>'+(fc>1?'<div class="face-tag visitor">+'+String(fc-1)+'</div>':'');
    lfn=fn;
  }else if(fc===0&&lfn!==''){$('face-bar').innerHTML='<span style="color:rgba(255,255,255,.2)">waiting for friends</span>';lfn='';}

  $('s-lang').textContent=s.current_lang||'-';

  $('c-rob').textContent=s.robot_online?'online':'offline';$('c-rob').style.color=s.robot_online?'#bbf7d0':'#fca5a5';
  $('c-st').textContent=s.anim_state;$('c-fc').textContent=fc;$('c-kn').textContent=s.known_person_count;
  $('c-lg').textContent=s.current_lang;$('c-tn').textContent=s.turn_count;
  $('c-up').textContent=Math.round(s.uptime_s)+'s';
  $('c-stt').textContent=(s.stt_s>0?s.stt_s.toFixed(2)+'s':'-');
  $('c-ttf').textContent=(s.llm_ttf_s>0?s.llm_ttf_s.toFixed(2)+'s':'-');
  $('c-tta').textContent=(s.tts_tta_s>0?s.tts_tta_s.toFixed(2)+'s':'-');
  $('c-prv').textContent=s.llm_provider;$('c-mod').textContent=s.llm_model||'-';
  $('c-ti').textContent=s.tokens_in;$('c-to').textContent=s.tokens_out;
  $('c-co').textContent='$'+s.est_cost_usd.toFixed(4);
  $('c-lu').textContent=s.last_user||'-';$('c-lr').textContent=s.last_reply||'-';

  kid=s.kid_mode;T('kid-t',kid);mute=s.muted;T('mute-t',mute);crowd=!!s.crowd_mode;T('crowd-t',crowd);
  if(!vd){$('vol-sl').value=s.volume;$('s-vol').textContent=s.volume.toFixed(1);}
  {const proj=s.audio_device==='projector';$('out-robot').classList.toggle('on',!proj);$('out-proj').classList.toggle('on',proj);}
  if(!rd){$('rate-sl').value=parseInt(s.speech_rate)||20;$('s-rate').textContent=s.speech_rate;}
  if(!ed){$('ene-sl').value=s.energy;$('s-ene').textContent=s.energy.toFixed(1);}
  if(!rmsd){$('rms-sl').value=s.gate_min_rms;$('s-rms').textContent=Math.round(s.gate_min_rms);}
  if(!void_){$('voi-sl').value=s.gate_min_voiced;$('s-voi').textContent=s.gate_min_voiced.toFixed(2);}
  if(!pkd){$('pk-sl').value=s.gate_min_peak;$('s-pk').textContent=s.gate_min_peak.toFixed(2);}
  if(!durd){$('dur-sl').value=s.gate_min_dur;$('s-dur').textContent=s.gate_min_dur.toFixed(2);}
  if(!vadd){$('vad-sl').value=s.vad_thresh;$('s-vad').textContent=s.vad_thresh.toFixed(2);}
  if(!brgd){$('brg-sl').value=s.barge_thresh;$('s-brg').textContent=s.barge_thresh.toFixed(2);}
  if(s.known_person_count!==lp){lp=s.known_person_count;H();}

  // ── Tech tab: pipeline strip, gate metrics, last-turn readout ──
  // (elements always exist in the DOM even when Tech isn't the active tab, so
  // updating them here unconditionally is a safe no-op the rest of the time.)
  $('p-mic-v').textContent=Math.round(s.mic_rms||0);
  const vadOn=!!s.vad_in_speech;
  $('p-vad').className='pill'+(vadOn?' on':'');
  $('p-vad-v').textContent='thr '+(typeof s.vad_thresh==='number'?s.vad_thresh.toFixed(2):'-');
  const gateOk=!!s.gate_ok;
  $('p-gate').className='pill'+(gateOk?' on':(s.gate_reason?' off':''));
  $('p-gate-v').textContent=gateOk?'OK':(s.gate_reason||'-');
  $('p-stt').className='pill'+(s.stt_s>0?' on':'');
  $('p-stt-v').textContent=s.stt_s>0?s.stt_s.toFixed(2)+'s':'-';
  $('p-llm').className='pill'+(s.llm_ttf_s>0?' on':'');
  $('p-llm-v').textContent=s.llm_ttf_s>0?s.llm_ttf_s.toFixed(2)+'s':'-';
  $('p-tts').className='pill'+(s.tts_tta_s>0?' on':'');
  $('p-tts-v').textContent=s.tts_tta_s>0?s.tts_tta_s.toFixed(2)+'s':'-';

  gateRow('g-rms','g-rms-c',s.gate_rms,s.gate_min_rms);
  gateRow('g-voi','g-voi-c',s.gate_voiced,s.gate_min_voiced);
  gateRow('g-pk','g-pk-c',s.gate_peak,s.gate_min_peak);
  gateRow('g-dur','g-dur-c',s.gate_dur,s.gate_min_dur);
  const gh=$('gate-headline');
  gh.textContent=gateOk?'PASSED':'REJECTED: '+(s.gate_reason||'unknown');
  gh.style.color=gateOk?'#bbf7d0':'#fecaca';

  $('t-lang').textContent=s.current_lang||'-';
  $('t-lu').textContent=s.last_user||'-';
  $('t-lr').textContent=s.last_reply||'-';
  $('t-stt').textContent=(s.stt_s>0?s.stt_s.toFixed(2)+'s':'-');
  $('t-ttf').textContent=(s.llm_ttf_s>0?s.llm_ttf_s.toFixed(2)+'s':'-');
  $('t-tta').textContent=(s.tts_tta_s>0?s.tts_tta_s.toFixed(2)+'s':'-');
  $('t-wait').textContent=(s.reply_wait_s>0?s.reply_wait_s.toFixed(2)+'s':'-');
  $('t-talk').textContent=(s.talk_s>0?s.talk_s.toFixed(2)+'s':'-');
}
function gateRow(vId,cId,measured,floor){
  const m=typeof measured==='number'?measured:0, f=typeof floor==='number'?floor:0;
  $(vId).textContent=m.toFixed(2)+' / '+f.toFixed(2);
  const pass=m>=f;
  const c=$(cId);c.textContent=pass?'PASS':'FAIL';c.className='chip '+(pass?'pass':'fail');
}
function esc(t){return String(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function T(id,on){$(id).classList.toggle('on',on);}
function P(url,body){fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined}).then(r=>r.json()).catch(()=>{});}
function S(){const t=$('say-i');if(t.value.trim()){P('/api/say',{text:t.value});t.value='';}}
$('kid-t').onclick=()=>{kid=!kid;T('kid-t',kid);P('/api/kid',{on:kid});};
$('mute-t').onclick=()=>{mute=!mute;T('mute-t',mute);P('/api/mute',{muted:mute});};
$('crowd-t').onclick=()=>{crowd=!crowd;T('crowd-t',crowd);P('/api/crowd',{on:crowd});};

function setOut(dev){
  $('out-robot').classList.toggle('on',dev==='robot');
  $('out-proj').classList.toggle('on',dev==='projector');
  P('/api/output',{device:dev});
}
let vt,rt,et;
function dV(k,v){
  if(k==='vol'){vd=true;$('s-vol').textContent=parseFloat(v).toFixed(1);clearTimeout(vt);vt=setTimeout(()=>{P('/api/volume',{volume:parseFloat(v)});vd=false;},300);}
  else if(k==='rate'){rd=true;const r=(v>=0?'+':'')+v+'%';$('s-rate').textContent=r;clearTimeout(rt);rt=setTimeout(()=>{P('/api/rate',{rate:r});rd=false;},300);}
  else{ed=true;$('s-ene').textContent=parseFloat(v).toFixed(1);clearTimeout(et);et=setTimeout(()=>{P('/api/energy',{energy:parseFloat(v)});ed=false;},300);}
}
['vol-sl','rate-sl','ene-sl'].forEach(id=>{const el=$(id);el.addEventListener('mousedown',()=>{if(id==='vol-sl')vd=true;if(id==='rate-sl')rd=true;if(id==='ene-sl')ed=true;});el.addEventListener('mouseup',()=>{vd=rd=ed=false;});});

let rmst,voit,pkt,durt,vadt,brgt;
function dA(k,v){
  if(k==='rms'){rmsd=true;$('s-rms').textContent=Math.round(v);clearTimeout(rmst);rmst=setTimeout(()=>{P('/api/audiotune',{min_rms:parseFloat(v)});rmsd=false;},300);}
  else if(k==='voi'){void_=true;$('s-voi').textContent=parseFloat(v).toFixed(2);clearTimeout(voit);voit=setTimeout(()=>{P('/api/audiotune',{min_voiced:parseFloat(v)});void_=false;},300);}
  else if(k==='pk'){pkd=true;$('s-pk').textContent=parseFloat(v).toFixed(2);clearTimeout(pkt);pkt=setTimeout(()=>{P('/api/audiotune',{min_peak:parseFloat(v)});pkd=false;},300);}
  else if(k==='dur'){durd=true;$('s-dur').textContent=parseFloat(v).toFixed(2);clearTimeout(durt);durt=setTimeout(()=>{P('/api/audiotune',{min_dur:parseFloat(v)});durd=false;},300);}
  else if(k==='vad'){vadd=true;$('s-vad').textContent=parseFloat(v).toFixed(2);clearTimeout(vadt);vadt=setTimeout(()=>{P('/api/audiotune',{vad_thresh:parseFloat(v)});vadd=false;},300);}
  else{brgd=true;$('s-brg').textContent=parseFloat(v).toFixed(2);clearTimeout(brgt);brgt=setTimeout(()=>{P('/api/audiotune',{barge_thresh:parseFloat(v)});brgd=false;},300);}
}
['rms-sl','voi-sl','pk-sl','dur-sl','vad-sl','brg-sl'].forEach(id=>{const el=$(id);el.addEventListener('mousedown',()=>{if(id==='rms-sl')rmsd=true;if(id==='voi-sl')void_=true;if(id==='pk-sl')pkd=true;if(id==='dur-sl')durd=true;if(id==='vad-sl')vadd=true;if(id==='brg-sl')brgd=true;});el.addEventListener('mouseup',()=>{rmsd=void_=pkd=durd=vadd=brgd=false;});});

const gg=$('g-grid');
G.forEach(g=>{const b=document.createElement('b');b.className='df';b.textContent=g;b.onclick=()=>P('/api/gesture',{name:g});gg.appendChild(b);});

// Quick-play phrases — pre-rendered to cached WAVs server-side so they fire
// instantly. Injected from reachy_demo/phrases.py (single source of truth, so
// the cache keys match). label → spoken text.
const PHRASES=__PHRASES__;
const pg=$('ph-grid');
PHRASES.forEach(([label,text])=>{const b=document.createElement('b');b.className='df';b.textContent=label;b.onclick=()=>P('/api/say',{text});pg.appendChild(b);});

let lp=-1;
function H(){
  fetch('/api/people').then(r=>r.json()).then(d=>{
    const p=$('ppl');
    if(!d.people||!d.people.length){p.innerHTML='<div style="color:#64748b">none enrolled</div>';return;}
    const grid=document.createElement('div');grid.className='pgrid';
    d.people.forEach(x=>{
      const card=document.createElement('div');card.className='pcard';
      const img=document.createElement('img');
      img.src='/api/face/'+encodeURIComponent(x.name)+'?t='+Date.now();
      img.alt='';
      img.onerror=()=>{const ni=document.createElement('div');ni.className='noimg';ni.textContent='🙂';img.replaceWith(ni);};
      const n=document.createElement('div');n.className='n';n.textContent=x.name;
      const f=document.createElement('div');f.className='f';
      f.textContent=(x.facts&&x.facts.length)?x.facts.length+' fact(s)':'no facts';
      const acts=document.createElement('div');acts.className='actions';
      const bf=document.createElement('button');bf.textContent='📝';bf.title='Edit facts';
      bf.onclick=()=>editFacts(x.name,x.facts||[]);
      const be=document.createElement('button');be.textContent='✏️';be.title='Rename';
      be.onclick=()=>renamePerson(x.name);
      const bd=document.createElement('button');bd.textContent='🗑️';bd.title='Delete';
      bd.onclick=()=>deletePerson(x.name);
      acts.appendChild(bf);acts.appendChild(be);acts.appendChild(bd);
      card.appendChild(img);card.appendChild(n);card.appendChild(f);card.appendChild(acts);
      grid.appendChild(card);
    });
    p.innerHTML='';p.appendChild(grid);
  }).catch(()=>{});
}
H();

function renamePerson(name){
  const next=prompt('Rename "'+name+'" to:',name);
  if(!next||!next.trim()||next.trim()===name)return;
  P('/api/person/rename',{old:name,new:next.trim()});
  setTimeout(H,400);
}
function deletePerson(name){
  if(!confirm('Delete "'+name+'"? This removes their photos and face data.'))return;
  P('/api/person/delete',{name:name});
  setTimeout(H,400);
}
function editFacts(name,facts){
  const cur=(facts||[]).join('\n');
  const next=prompt('Facts about "'+name+'" (one per line):',cur);
  if(next===null)return;               // cancelled
  const list=next.split('\n').map(s=>s.trim()).filter(Boolean);
  P('/api/person/facts',{name:name,facts:list});
  setTimeout(H,400);
}

fetch('/api/dances').then(r=>r.json()).then(d=>{
  const db=$('d-grid');
  (d.dances||[]).forEach(dn=>{
    const b=document.createElement('b');b.className='warn';b.textContent=dn.label;b.onclick=()=>P('/api/dance',{name:dn.key});db.appendChild(b);
  });
}).catch(()=>{});
</script>
</body>
</html>"""


async def _safe_body(request: Request) -> dict:
    """Parse a JSON request body, returning {} on any error or non-object.

    Keeps a malformed POST (kid mashing a keyboard, a stray curl) from 500-ing
    a control endpoint mid-demo instead of being a harmless no-op.
    """
    try:
        b = await request.json()
        return b if isinstance(b, dict) else {}
    except Exception:
        return {}


def _safe_float(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class WebStage:
    def __init__(self, state: LiveState, camera_hub: CameraHub,
                 fid: "FaceIdentifier | None" = None,
                 host: str = "0.0.0.0", port: int = 8080) -> None:
        self.state = state
        self.camera_hub = camera_hub
        self.fid = fid  # FaceIdentifier — powers gallery photos + delete/rename
        self.host = host
        self.port = port
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self._server = None  # uvicorn.Server, set in start()
        self.app = FastAPI(title="Reachy")
        self._register_routes()

    def _register_routes(self) -> None:
        gestures_json = json.dumps(GESTURES)
        from reachy_demo.phrases import QUICK_PHRASES
        phrases_json = json.dumps(QUICK_PHRASES)

        @self.app.get("/")
        def _index() -> HTMLResponse:
            return HTMLResponse(_HTML.replace("__GESTURES__", gestures_json)
                                     .replace("__PHRASES__", phrases_json))

        @self.app.get("/video")
        def _video() -> StreamingResponse:
            return StreamingResponse(self._video_stream(),
                                     media_type="multipart/x-mixed-replace; boundary=frame")

        @self.app.get("/status")
        def _status() -> dict:
            return self.state.snapshot()

        @self.app.websocket("/ws")
        async def _ws(websocket: WebSocket) -> None:
            await websocket.accept()
            last_key, last_sent = None, 0.0
            try:
                while not self._stop_flag.is_set():
                    snap = self.state.snapshot()
                    key = json.dumps({k: v for k, v in snap.items() if k != "uptime_s"})
                    now = time.time()
                    if key != last_key or now - last_sent >= 1.0:
                        await websocket.send_text(json.dumps(snap))
                        last_key = key; last_sent = now
                    await asyncio.sleep(0.12)
            except WebSocketDisconnect: pass
            except Exception: pass

        @self.app.websocket("/scope")
        async def _scope(websocket: WebSocket) -> None:
            # High-rate (~30 Hz) mic-level stream for the Tech tab's oscilloscope —
            # the main /ws is throttled to ~8 Hz which is too coarse to draw a
            # smooth scope trace. Keep this endpoint minimal so a stalled scope
            # client never affects the main dashboard websocket.
            await websocket.accept()
            try:
                while not self._stop_flag.is_set():
                    await websocket.send_text(json.dumps({
                        "rms": self.state.mic_rms,
                        "floor": self.state.gate_min_rms,
                        "in_speech": self.state.vad_in_speech,
                        "vad": self.state.vad_thresh,
                    }))
                    await asyncio.sleep(0.033)
            except WebSocketDisconnect: pass
            except Exception: pass

        @self.app.post("/api/wake")
        def _w() -> dict: self.state.request_wake(); return {"ok": True}
        @self.app.post("/api/sleep")
        def _sl() -> dict: self.state.request_sleep(); return {"ok": True}
        @self.app.post("/api/say")
        async def _sy(request: Request) -> dict:
            d = await _safe_body(request); self.state.request_say(str(d.get("text", ""))[:500]); return {"ok": True}
        @self.app.post("/api/stop")
        def _st() -> dict: self.state.request_shutdown(); return {"ok": True}
        @self.app.post("/api/mute")
        async def _mu(request: Request) -> dict:
            d = await _safe_body(request); self.state.muted = bool(d.get("muted", False)); return {"ok": True}
        @self.app.post("/api/kid")
        async def _ki(request: Request) -> dict:
            d = await _safe_body(request); self.state.kid_mode = bool(d.get("on", True)); return {"ok": True}
        @self.app.post("/api/gesture")
        async def _ge(request: Request) -> dict:
            d = await _safe_body(request); self.state.request_gesture(str(d.get("name", ""))); return {"ok": True}
        @self.app.post("/api/volume")
        async def _vo(request: Request) -> dict:
            d = await _safe_body(request); self.state.volume = max(0.0, min(5.0, _safe_float(d.get("volume"), 2.5))); return {"ok": True}
        @self.app.post("/api/output")
        async def _op(request: Request) -> dict:
            d = await _safe_body(request)
            self.state.audio_device = "projector" if str(d.get("device")) == "projector" else "robot"
            return {"ok": True, "device": self.state.audio_device}
        @self.app.post("/api/crowd")
        async def _cr(request: Request) -> dict:
            d = await _safe_body(request); self.state.crowd_mode = bool(d.get("on", False)); return {"ok": True}
        @self.app.post("/api/rate")
        async def _ra(request: Request) -> dict:
            # Validate the edge-tts rate string ("+NN%") before it reaches
            # tts_edge.RATE — a malformed value would break every TTS call for
            # the rest of the session with no self-recovery. Ignore bad input.
            d = await _safe_body(request)
            rate = str(d.get("rate", "+20%")).strip()[:10]
            if re.fullmatch(r"[+-]?\d{1,3}%", rate):
                self.state.speech_rate = rate
            return {"ok": True, "rate": self.state.speech_rate}
        @self.app.post("/api/energy")
        async def _en(request: Request) -> dict:
            d = await _safe_body(request); self.state.energy = max(0.0, min(1.0, _safe_float(d.get("energy"), 1.0))); return {"ok": True}
        @self.app.post("/api/audiotune")
        async def _at(request: Request) -> dict:
            d = await _safe_body(request)
            if "min_rms" in d:
                self.state.gate_min_rms = max(0.0, min(2000.0, _safe_float(d.get("min_rms"), self.state.gate_min_rms)))
            if "min_voiced" in d:
                self.state.gate_min_voiced = max(0.0, min(1.0, _safe_float(d.get("min_voiced"), self.state.gate_min_voiced)))
            if "min_peak" in d:
                self.state.gate_min_peak = max(0.0, min(1.0, _safe_float(d.get("min_peak"), self.state.gate_min_peak)))
            if "min_dur" in d:
                self.state.gate_min_dur = max(0.0, min(1.5, _safe_float(d.get("min_dur"), self.state.gate_min_dur)))
            if "vad_thresh" in d:
                self.state.vad_thresh = max(0.1, min(0.95, _safe_float(d.get("vad_thresh"), self.state.vad_thresh)))
            if "barge_thresh" in d:
                self.state.barge_thresh = max(0.1, min(0.95, _safe_float(d.get("barge_thresh"), self.state.barge_thresh)))
            return {"ok": True}
        @self.app.post("/api/dance")
        async def _da(request: Request) -> dict:
            d = await _safe_body(request); self.state.request_dance(str(d.get("name", "macarena"))); return {"ok": True}
        @self.app.get("/api/dances")
        def _ds() -> dict:
            from reachy_demo.dance import DANCES
            return {"dances": [{"key": k, "label": v["label"], "bpm": v["bpm"], "duration": v["duration"]} for k, v in DANCES.items()]}
        @self.app.get("/api/people")
        def _pe() -> dict:
            from reachy_demo.memory import known_people, load_person_facts
            # Prefer the face-id roster (has photos) over memory.known_people()
            # so the gallery names resolve to a faces/<slug>/ directory.
            if self.fid is not None and self.fid._ref_names:
                names = sorted(set(self.fid._ref_names))
            else:
                names = known_people()
            return {"people": [{"name": n, "facts": load_person_facts(n)} for n in names]}

        @self.app.get("/api/face/{name}")
        def _face(name: str):
            slug = _face_slug(name)
            if not slug:
                raise HTTPException(status_code=404, detail="not found")
            faces_root = (self.fid.faces_dir if self.fid is not None else _DEFAULT_FACES_DIR).resolve()
            pdir = (faces_root / slug).resolve()
            if pdir != faces_root and faces_root not in pdir.parents:
                raise HTTPException(status_code=400, detail="invalid name")
            if not pdir.is_dir():
                raise HTTPException(status_code=404, detail="not found")
            photos = []
            for pat in ("*.jpg", "*.jpeg", "*.png"):
                photos.extend(pdir.glob(pat))
            if not photos:
                raise HTTPException(status_code=404, detail="no photo")
            photos.sort(key=lambda p: p.stat().st_mtime)
            return FileResponse(str(photos[-1]))

        @self.app.post("/api/person/delete")
        async def _pdel(request: Request) -> dict:
            d = await _safe_body(request)
            name = str(d.get("name", "")).strip()[:80]
            if not name or name.lower() == "visitor":
                return {"ok": False, "error": "invalid name"}
            if self.fid is None:
                return {"ok": False, "error": "face id not available"}
            removed = self.fid.delete_person(name)
            from reachy_demo.memory import delete_person_memory
            delete_person_memory(name)
            self.state.known_person_count = (
                len(set(self.fid._ref_names)) if self.fid._ref_names else 0)
            return {"ok": removed > 0, "removed": removed}

        @self.app.post("/api/person/rename")
        async def _pren(request: Request) -> dict:
            d = await _safe_body(request)
            old = str(d.get("old", "")).strip()[:80]
            new = str(d.get("new", "")).strip()[:80]
            if not old or not new or new.lower() == "visitor":
                return {"ok": False, "error": "invalid name"}
            if self.fid is None:
                return {"ok": False, "error": "face id not available"}
            renamed = self.fid.rename_person(old, new)
            from reachy_demo.memory import rename_person_facts
            rename_person_facts(old, new)
            self.state.known_person_count = (
                len(set(self.fid._ref_names)) if self.fid._ref_names else 0)
            return {"ok": renamed > 0, "renamed": renamed}

        @self.app.post("/api/person/facts")
        async def _pfacts(request: Request) -> dict:
            d = await _safe_body(request)
            name = str(d.get("name", "")).strip()[:80]
            facts = d.get("facts", [])
            if not name or name.lower() == "visitor" or not isinstance(facts, list):
                return {"ok": False, "error": "invalid"}
            from reachy_demo.memory import set_person_facts
            ok = set_person_facts(name, [str(f)[:200] for f in facts][:20])
            return {"ok": ok}

    def _video_stream(self):
        heartbeat = 0
        while not self._stop_flag.is_set():
            try: jpg = self.camera_hub.mjpeg_bytes()
            except Exception: jpg = None
            if jpg is not None:
                heartbeat = 0
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            else:
                heartbeat += 1
                if heartbeat >= 60:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 0\r\n\r\n\r\n"
                    heartbeat = 0
            time.sleep(1 / 30)

    def start(self) -> None:
        import uvicorn
        self._stop_flag.clear()
        # Hold the Server instance (not the fire-and-forget uvicorn.run) so
        # stop() can actually free port 8080 — otherwise a supervised restart
        # of the demo can't rebind the dashboard and the projector goes dark.
        config = uvicorn.Config(self.app, host=self.host, port=self.port,
                                log_level="warning")
        self._server = uvicorn.Server(config)

        def _serve() -> None:
            try:
                self._server.run()
            except Exception as e:  # e.g. port 8080 already held by an orphan run
                print(f"\n*** DASHBOARD FAILED TO START on {self.host}:{self.port}: {e}\n"
                      f"*** The projector/control page will not load. Is another demo "
                      f"still running? Try: pkill -9 -f reachy-mini-daemon\n", flush=True)

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)
