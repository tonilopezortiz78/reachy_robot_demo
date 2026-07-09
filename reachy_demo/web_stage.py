"""
reachy_demo/web_stage.py — tabbed web dashboard for the live kids demo.

One website, two tabs, any screen can open either:

    /          — single page with a tab bar:
      #stage    — big, bold, kid-facing view for the PROJECTOR. Camera feed,
                  "Reachy hears / thinks / says" live captions, gesture word,
                  state badge. Pure spectacle, no controls.
      #control  — operator CONTROL PANEL for the laptop. Wake/sleep/stop,
                  "make Reachy say this" puppet box, 19 gesture buttons,
                  Macarena trigger, kid-mode + mute toggles, latency/cost,
                  roster. Your steering wheel during the show.

    Each screen opens the same URL and clicks its tab (or uses the hash:
    http://<ip>:8080/#stage on the projector, /#control on the laptop).
    The chosen tab is saved in localStorage so each browser remembers its view.

Shared infrastructure (same patterns as web_server.py):
    GET  /video   — MJPEG live stream
    GET  /status  — JSON snapshot (fallback for when WS is down)
    WS   /ws      — pushes LiveState snapshot ~8-10x/s on change
    POST /api/wake | /api/sleep | /api/say | /api/stop | /api/mute
    POST /api/gesture | /api/dance | /api/kid
    GET  /api/people — roster from reachy_demo.memory
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


GESTURES = [
    "acknowledge", "yes", "no", "thank", "thinking", "curious", "confused",
    "greeting", "celebrate", "proud", "amazed", "love", "laugh", "oops",
    "shy", "surprised", "cheerful", "success", "relief",
]

GESTURE_EMOJI = {
    "acknowledge": "ok", "yes": "YES", "no": "NO", "thank": "TY",
    "thinking": "hmm", "curious": "see", "confused": "?", "greeting": "HI",
    "celebrate": "YAY", "proud": "proud", "amazed": "whoa", "love": "love",
    "laugh": "lol", "oops": "oops", "shy": "shy", "surprised": "!",
    "cheerful": "yay", "success": "WIN", "relief": "phew", "": "",
}

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reachy</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;background:#0a0a0a;color:#fff;font-family:'Segoe UI',system-ui,sans-serif;overflow:hidden}

/* ── tab bar ── */
#tabs{position:fixed;top:0;left:0;right:0;z-index:100;display:flex;gap:0;background:#0f172a;border-bottom:2px solid #1e293b}
#tabs button{flex:1;max-width:200px;padding:10px 20px;background:transparent;border:none;color:#64748b;font-size:.9rem;font-weight:700;cursor:pointer;letter-spacing:1px;border-bottom:3px solid transparent;transition:.15s}
#tabs button:hover{color:#94a3b8}
#tabs button.active{color:#fff;border-bottom-color:#3b82f6;background:#111827}

/* ── view switching ── */
.view{display:none;height:100vh;padding-top:48px}
.view.active{display:block}

/* ════════════════════════════════════════════════════════════════
   STAGE VIEW (projector) — terminal-style live console
   ════════════════════════════════════════════════════════════════ */
#stage-view{display:none}
#stage-view.active{display:grid;grid-template-columns:1.5fr 1fr;grid-template-rows:1fr auto;gap:10px;padding:54px 10px 10px;transition:background-color .6s ease}
#stage-view.speaking{background:#0c1a2e}
#stage-view.listening{background:#0a1f0e}
#stage-view.thinking{background:#1f1408}
#stage-view.idle{background:#0a0a0a}
#cam-stage{grid-row:1/3;position:relative;background:#000;border-radius:12px;overflow:hidden;border:2px solid #1f2937}
#cam-stage img{width:100%;height:100%;object-fit:cover}
#tracker-bar{position:absolute;bottom:0;left:0;right:0;display:flex;gap:6px;padding:8px;background:rgba(0,0,0,.75);overflow-x:auto;min-height:42px;align-items:center}
.face-tag{display:flex;align-items:center;gap:6px;padding:5px 12px;border-radius:999px;font-size:1rem;font-weight:700;white-space:nowrap;flex-shrink:0;backdrop-filter:blur(4px)}
.face-tag.known{background:rgba(34,197,94,.25);border:1px solid rgba(34,197,94,.5);color:#bbf7d0}
.face-tag.visitor{background:rgba(59,130,246,.25);border:1px solid rgba(59,130,246,.5);color:#bfdbfe}
.face-tag .conf{font-size:.7rem;opacity:.7;font-weight:400}
#state-badge{position:absolute;top:12px;left:12px;padding:6px 16px;border-radius:999px;font-size:1.1rem;font-weight:800;letter-spacing:2px;backdrop-filter:blur(4px);background:rgba(0,0,0,.5)}
#gesture-emoji{position:absolute;top:12px;right:12px;font-size:2rem;font-weight:800;text-shadow:0 0 14px rgba(0,0,0,.9);color:#fbbf24}
/* right: terminal console */
#con{display:flex;flex-direction:column;gap:6px;overflow:hidden}
#con-log{flex:1;overflow-y:auto;font-family:'SF Mono','Cascadia Code','Consolas',monospace;font-size:1.1rem;line-height:1.5;padding:8px 12px;background:#0c0c0c;border-radius:10px;border:2px solid #1f2937;min-height:0}
.log-line{padding:3px 0;border-bottom:1px solid #1a1a1a;animation:fk .3s}
@keyframes fk{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:translateY(0)}}
.log-line .time{color:#374151;margin-right:8px;font-size:.7rem}
.log-line .glyph{font-size:1.1rem;margin-right:6px}
.log-line.listening{color:#86efac}.log-line.heard{color:#93c5fd}.log-line.thinking{color:#fde68a}.log-line.speaking{color:#93c5fd;font-weight:700}.log-line.gesture{color:#fbbf24}.log-line.done{color:#64748b}
#con-current{flex-shrink:0;padding:10px 14px;background:#111827;border-radius:10px;border:2px solid #1f2937;font-size:1.4rem;font-weight:700;min-height:2.8em;display:flex;align-items:center;gap:10px}
#con-current .glyph{font-size:1.6rem}
#con-current .blink{display:inline-block;width:4px;height:1.8rem;background:currentColor;margin-left:4px;animation:bk 1s steps(2) infinite}
@keyframes bk{50%{opacity:0}}
#stage-footer{grid-column:2;font-size:.7rem;color:#374151;text-align:right}

/* ════════════════════════════════════════════════════════════════
   CONTROL VIEW (operator laptop)
   ════════════════════════════════════════════════════════════════ */
#control-view.active{display:block;overflow:auto;padding:60px 14px 14px}
.ctrl-h1{font-size:1.1rem;color:#fff;margin-bottom:12px}
.ctrl-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;max-width:1400px}
.card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px}
.card h2{font-size:.72rem;text-transform:uppercase;letter-spacing:1.5px;color:#94a3b8;margin-bottom:10px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px}
button.ctrl{background:#334155;color:#e5e7eb;border:1px solid #475569;border-radius:6px;padding:7px 12px;cursor:pointer;font-size:.82rem;font-weight:600;transition:.12s}
button.ctrl:hover{background:#475569;border-color:#64748b}
button.ctrl:active{transform:translateY(1px)}
button.ctrl.danger{background:#7f1d1d;border-color:#991b1b;color:#fecaca}
button.ctrl.danger:hover{background:#991b1b}
button.ctrl.go{background:#14532d;border-color:#166534;color:#bbf7d0}
button.ctrl.go:hover{background:#166534}
button.ctrl.warn{background:#78350f;border-color:#92400e;color:#fde68a}
button.ctrl.warn:hover{background:#92400e}
input[type=text]{background:#0f172a;border:1px solid #475569;color:#fff;border-radius:6px;padding:8px 10px;font-size:.9rem;flex:1;min-width:120px}
input[type=text]:focus{outline:none;border-color:#60a5fa}
.ggrid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
.ggrid button{padding:6px 4px;font-size:.75rem;text-align:center}
.stat{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #334155;font-size:.82rem}
.stat:last-child{border-bottom:none}
.stat .k{color:#94a3b8}.stat .v{color:#fff;font-weight:600;font-variant-numeric:tabular-nums}
.toggle{display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none}
.toggle .sw{width:38px;height:22px;background:#334155;border-radius:999px;position:relative;transition:.2s}
.toggle .sw::after{content:'';position:absolute;top:2px;left:2px;width:18px;height:18px;background:#fff;border-radius:50%;transition:.2s}
.toggle.on .sw{background:#22c55e}.toggle.on .sw::after{left:18px}
.people{max-height:180px;overflow:auto}
.person{padding:6px 8px;border-bottom:1px solid #334155}
.person .n{font-weight:700;color:#fff}.person .f{font-size:.75rem;color:#94a3b8}
.cam-prev{width:100%;border-radius:8px;border:1px solid #334155;margin-bottom:8px}
</style>
</head>
<body>
<div id="tabs">
  <button id="tab-stage" onclick="showTab('stage')">Stage</button>
  <button id="tab-control" onclick="showTab('control')">Control</button>
</div>

<!-- ═══ STAGE VIEW ═══ -->
<div id="stage-view" class="view">
  <div id="cam-stage">
    <img id="feed-stage" src="/video" alt="camera" onerror="reconnect('feed-stage')">
    <div id="state-badge">IDLE</div>
    <div id="gesture-emoji"></div>
    <div id="tracker-bar"><span style="color:#64748b;font-size:.85rem">no faces</span></div>
  </div>
  <div id="con">
    <div id="con-log"></div>
    <div id="con-current"><span style="color:#64748b">...</span></div>
  </div>
  <div id="stage-footer">NS Reachy Mini · <span id="s-lang-stage">-</span> · <span id="s-face-count">0</span> faces</div>
</div>

<!-- ═══ CONTROL VIEW ═══ -->
<div id="control-view" class="view">
  <h1 class="ctrl-h1">Reachy Control Panel</h1>
  <div class="ctrl-grid">
    <div class="card">
      <h2>Commands</h2>
      <div class="row">
        <button class="ctrl go" onclick="post('/api/wake')">Wake</button>
        <button class="ctrl warn" onclick="post('/api/sleep')">Sleep</button>
        <button class="ctrl danger" onclick="post('/api/stop')">Stop demo</button>
      </div>
      <div class="row">
        <label class="toggle" id="kid-toggle"><div class="sw"></div><span>Kid mode</span></label>
        <label class="toggle" id="mute-toggle"><div class="sw"></div><span>Mute</span></label>
      </div>
      <h2 style="margin-top:14px">Make Reachy say</h2>
      <div class="row">
        <input type="text" id="say-text" placeholder="Type a line, Enter to fire" onkeydown="if(event.key==='Enter')doSay()">
        <button class="ctrl go" onclick="doSay()">Say</button>
      </div>
    </div>
    <div class="card">
      <h2>Gestures (19) - tap to trigger</h2>
      <div class="ggrid" id="gesture-grid"></div>
      <h2 style="margin-top:14px">Dances</h2>
      <div class="row" id="dance-buttons"></div>
    </div>
    <div class="card">
      <h2>Camera</h2>
      <img class="cam-prev" src="/video" onerror="this.src='/video?t='+Date.now()">
      <h2>Status</h2>
      <div class="stat"><span class="k">Robot</span><span class="v" id="s-robot">-</span></div>
      <div class="stat"><span class="k">State</span><span class="v" id="s-state">-</span></div>
      <div class="stat"><span class="k">Gesture</span><span class="v" id="s-gesture">-</span></div>
      <div class="stat"><span class="k">Faces</span><span class="v" id="s-faces">0</span></div>
      <div class="stat"><span class="k">Known</span><span class="v" id="s-known">0</span></div>
      <div class="stat"><span class="k">Lang</span><span class="v" id="s-lang">-</span></div>
      <div class="stat"><span class="k">Turn</span><span class="v" id="s-turn">0</span></div>
      <div class="stat"><span class="k">Uptime</span><span class="v" id="s-up">0s</span></div>
    </div>
    <div class="card">
      <h2>Latency (last turn)</h2>
      <div class="stat"><span class="k">STT</span><span class="v" id="s-stt">-</span></div>
      <div class="stat"><span class="k">LLM TTF</span><span class="v" id="s-ttf">-</span></div>
      <div class="stat"><span class="k">TTS TTA</span><span class="v" id="s-tta">-</span></div>
      <div class="stat"><span class="k">Provider</span><span class="v" id="s-prov">-</span></div>
      <div class="stat"><span class="k">Model</span><span class="v" id="s-model">-</span></div>
    </div>
    <div class="card">
      <h2>Tokens and cost</h2>
      <div class="stat"><span class="k">Tokens in</span><span class="v" id="s-tin">0</span></div>
      <div class="stat"><span class="k">Tokens out</span><span class="v" id="s-tout">0</span></div>
      <div class="stat"><span class="k">Est. cost</span><span class="v" id="s-cost">$0.00</span></div>
      <div class="stat"><span class="k">Last user</span><span class="v" id="s-user" style="font-size:.75rem;text-align:right;max-width:60%">-</span></div>
      <div class="stat"><span class="k">Last reply</span><span class="v" id="s-reply" style="font-size:.75rem;text-align:right;max-width:60%">-</span></div>
    </div>
    <div class="card">
      <h2>Enrolled people</h2>
      <div class="people" id="people">loading...</div>
    </div>
    <div class="card">
      <h2>Audio and energy</h2>
      <div class="stat"><span class="k">Volume</span><span class="v" id="s-vol">2.5</span></div>
      <input type="range" id="vol-slider" min="0" max="5" step="0.1" value="2.5" style="width:100%;margin-bottom:10px" oninput="debVol(this.value)">
      <div class="stat"><span class="k">Speech rate</span><span class="v" id="s-rate">+20%</span></div>
      <input type="range" id="rate-slider" min="-30" max="50" step="5" value="20" style="width:100%;margin-bottom:10px" oninput="debRate(this.value)">
      <div class="stat"><span class="k">Antenna energy</span><span class="v" id="s-energy">1.0</span></div>
      <input type="range" id="energy-slider" min="0" max="1" step="0.1" value="1" style="width:100%">
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const G_E=__G_E__;
const GESTURES=__GESTURES__;
const STATE_C={idle:'#4b5563',listening:'#22c55e',thinking:'#f59e0b',speaking:'#3b82f6'};
const STATE_L={idle:'IDLE',listening:'LISTENING',thinking:'THINKING',speaking:'SPEAKING'};

/* ── tab switching ── */
function showTab(name){
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  $('stage-view').classList.toggle('active',name==='stage');
  $('control-view').classList.toggle('active',name==='control');
  $('tab-stage').classList.toggle('active',name==='stage');
  $('tab-control').classList.toggle('active',name==='control');
  localStorage.setItem('reachy-tab',name);
}
function initTab(){
  const hash=location.hash.replace('#','');
  const saved=localStorage.getItem('reachy-tab')||'control';
  showTab(hash==='stage'||hash==='control'?hash:saved);
}
window.addEventListener('hashchange',initTab);
initTab();

/* ── camera reconnect ── */
function reconnect(id){const el=$(id);setTimeout(()=>{el.src='/video?t='+Date.now();},1500);}

/* ── websocket ── */
let ws=null;
function connect(){
  ws=new WebSocket('ws://'+location.host+'/ws');
  ws.onmessage=e=>render(JSON.parse(e.data));
  ws.onclose=()=>setTimeout(connect,1500);
  ws.onerror=()=>ws.close();
}
connect();

/* ── render ── */
let lastAnim='idle',lastUser='',lastGesture='',lastPartial='',lastSpeech='',lastFaceName='-',lastFaces=0;
function esc(t){return String(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function fmt(s){if(!s||s<0)return'-';return s.toFixed(2)+'s';}
function now(){const d=new Date();return d.getHours().toString().padStart(2,'0')+':'+d.getMinutes().toString().padStart(2,'0')+':'+d.getSeconds().toString().padStart(2,'0');}
function logLine(cls,glyph,text){const l=document.createElement('div');l.className='log-line '+cls;l.innerHTML='<span class="time">'+now()+'</span><span class="glyph">'+glyph+'</span>'+esc(text);const con=$('con-log');con.appendChild(l);con.scrollTop=con.scrollHeight;if(con.children.length>80)con.firstChild.remove();}
function render(s){
  const st=s.anim_state||'idle';
  const sv=$('stage-view');
  sv.className='view'+(sv.classList.contains('active')?' active':'')+' '+st;
  // state badge
  const badge=$('state-badge');
  badge.textContent=STATE_L[st]||st.toUpperCase();
  badge.style.background=STATE_C[st]||'#4b5563';
  badge.className=(st==='speaking'||st==='listening')?'pulse':'';
  // gesture emoji
  const g=s.current_gesture||'';
  $('gesture-emoji').textContent=G_E[g]||'';
  // state transition log
  if(st!==lastAnim){
    const em={idle:'Zz',listening:'mic',thinking:'brain',speaking:'spk'};
    logLine(st,em[st]||'.',STATE_L[st]||st);
    lastAnim=st;
  }
  // user speech log
  if(s.last_user&&s.last_user!==lastUser){
    logLine('heard','ear',s.last_user);
    lastUser=s.last_user;
  }
  // current caption: speech > thinking > listening
  const speech=s.current_speech||'';
  const partial=s.llm_partial||'';
  const cur=$('con-current');
  if(speech&&speech!==lastSpeech){
    cur.innerHTML='<span style="color:#93c5fd;font-size:1.5rem">speak</span>'+esc(speech)+'<span class="blink" style="background:#93c5fd"></span>';
    lastSpeech=speech;
  }else if(!speech&&st==='thinking'&&partial&&partial!==lastPartial){
    cur.innerHTML='<span style="color:#fde68a;font-size:1.5rem">think</span><span style="color:#fbbf24;font-style:italic">\u201c'+esc(partial.slice(-80))+'\u201d</span>';
    lastPartial=partial;
  }else if(!speech&&st==='listening'){
    cur.innerHTML='<span style="color:#86efac;font-size:1.5rem">listen</span><span style="color:#86efac">Waiting for you...</span>';
    lastSpeech='';lastPartial='';
  }else if(!st||st==='idle'){
    cur.innerHTML='<span style="color:#64748b;font-size:1.5rem">Zz</span><span style="color:#64748b">Ready.</span>';
    lastSpeech='';lastPartial='';
  }
  // face tracker bar
  const fc=s.faces_visible||0;
  $('s-face-count').textContent=fc;
  if(fc>0&&(fc!==lastFaces||s.last_face_name!==lastFaceName)){
    const bar=$('tracker-bar');
    let html='';
    if(s.last_face_name&&s.last_face_name!=='-'&&s.last_face_conf>0.45){
      html+='<div class="face-tag known">'+esc(s.last_face_name)+'<span class="conf">'+Math.round(s.last_face_conf*100)+'%</span></div>';
    }
    if(fc>1){html+='<div class="face-tag visitor">+'+String(fc-1)+' more</div>';}
    else if(!html&&fc>0){html+='<div class="face-tag visitor">'+fc+' face(s)</div>';}
    bar.innerHTML=html||'<span style="color:#64748b;font-size:.85rem">tracking...</span>';
    lastFaces=fc;lastFaceName=s.last_face_name||'-';
  }
  // lang + faces footer
  $('s-lang-stage').textContent=s.current_lang||'-';
  // === control panel fields ===
  $('s-robot').textContent=s.robot_online?'online':'offline';
  $('s-robot').style.color=s.robot_online?'#bbf7d0':'#fca5a5';
  $('s-state').textContent=s.anim_state;
  $('s-gesture').textContent=s.current_gesture||'-';
  $('s-faces').textContent=s.faces_visible;
  $('s-known').textContent=s.known_person_count;
  $('s-lang').textContent=s.current_lang;
  $('s-turn').textContent=s.turn_count;
  $('s-up').textContent=Math.round(s.uptime_s)+'s';
  $('s-stt').textContent=fmt(s.stt_s);
  $('s-ttf').textContent=fmt(s.llm_ttf_s);
  $('s-tta').textContent=fmt(s.tts_tta_s);
  $('s-prov').textContent=s.llm_provider;
  $('s-model').textContent=s.llm_model||'-';
  $('s-tin').textContent=s.tokens_in;
  $('s-tout').textContent=s.tokens_out;
  $('s-cost').textContent='$'+s.est_cost_usd.toFixed(4);
  $('s-user').textContent=s.last_user||'-';
  $('s-reply').textContent=s.last_reply||'-';
  setToggle($('kid-toggle'),s.kid_mode);
  setToggle($('mute-toggle'),s.muted);
  // sliders (only update from state if user isn't dragging)
  if(!volDragging){$('vol-slider').value=s.volume;$('s-vol').textContent=s.volume.toFixed(1);}
  if(!rateDragging){$('rate-slider').value=parseInt(s.speech_rate)||20;$('s-rate').textContent=s.speech_rate;}
  if(!energyDragging){$('energy-slider').value=s.energy;$('s-energy').textContent=s.energy.toFixed(1);}
  // people roster refetch when count changes
  if(s.known_person_count!==lastPeopleCount){lastPeopleCount=s.known_person_count;fetchPeople();}
}

/* ── control actions ── */
function post(url,body){fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined}).then(r=>r.json()).catch(()=>{});}
function doSay(){const t=$('say-text');if(t.value.trim()){post('/api/say',{text:t.value});t.value='';}}
function setToggle(el,on){el.classList.toggle('on',on);}
let kidOn=true,muteOn=false;
$('kid-toggle').onclick=()=>{kidOn=!kidOn;post('/api/kid',{on:kidOn});setToggle($('kid-toggle'),kidOn);};
$('mute-toggle').onclick=()=>{muteOn=!muteOn;post('/api/mute',{muted:muteOn});setToggle($('mute-toggle'),muteOn);};

/* ── sliders with debounce ── */
let volDragging=false,rateDragging=false,energyDragging=false;
let volTimer=null,rateTimer=null,energyTimer=null;
function debVol(v){volDragging=true;$('s-vol').textContent=parseFloat(v).toFixed(1);clearTimeout(volTimer);volTimer=setTimeout(()=>{post('/api/volume',{volume:parseFloat(v)});volDragging=false;},300);}
function debRate(v){rateDragging=true;const r=(v>=0?'+':'')+v+'%';$('s-rate').textContent=r;clearTimeout(rateTimer);rateTimer=setTimeout(()=>{post('/api/rate',{rate:r});rateDragging=false;},300);}
function debEnergy(v){energyDragging=true;$('s-energy').textContent=parseFloat(v).toFixed(1);clearTimeout(energyTimer);energyTimer=setTimeout(()=>{post('/api/energy',{energy:parseFloat(v)});energyDragging=false;},300);}
$('vol-slider').addEventListener('mousedown',()=>volDragging=true);
$('vol-slider').addEventListener('mouseup',()=>volDragging=false);
$('rate-slider').addEventListener('mousedown',()=>rateDragging=true);
$('rate-slider').addEventListener('mouseup',()=>rateDragging=false);
$('energy-slider').addEventListener('mousedown',()=>energyDragging=true);
$('energy-slider').addEventListener('mouseup',()=>energyDragging=false);

/* ── people roster (refetched via WS when count changes) ── */
let lastPeopleCount=-1;
function fetchPeople(){
  fetch('/api/people').then(r=>r.json()).then(d=>{
    const p=$('people');
    if(!d.people||!d.people.length){p.innerHTML='<div style="color:#64748b">none enrolled</div>';return;}
    p.innerHTML=d.people.map(x=>'<div class="person"><div class="n">'+esc(x.name)+'</div><div class="f">'+(x.facts&&x.facts.length?x.facts.length+' fact(s)':'no facts')+'</div></div>').join('');
  }).catch(()=>{});
}
fetchPeople();

/* ── gesture grid ── */
const gg=$('gesture-grid');
GESTURES.forEach(g=>{const b=document.createElement('button');b.className='ctrl';b.textContent=g;b.onclick=()=>post('/api/gesture',{name:g});gg.appendChild(b);});

/* ── dance buttons ── */
fetch('/api/dances').then(r=>r.json()).then(d=>{
  const db=$('dance-buttons');
  (d.dances||[]).forEach(dn=>{
    const b=document.createElement('button');b.className='ctrl warn';
    b.textContent=dn.label+' ('+dn.bpm+' BPM)';
    b.onclick=()=>post('/api/dance',{name:dn.key});
    db.appendChild(b);
  });
}).catch(()=>{});

</script>
</body>
</html>"""


class WebStage:
    """Tabbed dashboard: one page, #stage and #control tabs."""

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
        self.app = FastAPI(title="Reachy Mini Stage")
        self._register_routes()

    def _register_routes(self) -> None:
        ge_json = json.dumps(GESTURE_EMOJI)
        gestures_json = json.dumps(GESTURES)

        @self.app.get("/")
        def _index() -> HTMLResponse:
            return HTMLResponse(
                _HTML.replace("__G_E__", ge_json)
                     .replace("__GESTURES__", gestures_json)
            )

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
            await websocket.accept()
            last_key: str | None = None
            last_sent = 0.0
            try:
                while not self._stop_flag.is_set():
                    snap = self.state.snapshot()
                    key = json.dumps({k: v for k, v in snap.items()
                                      if k != "uptime_s"})
                    now = time.time()
                    if key != last_key or now - last_sent >= 1.0:
                        await websocket.send_text(json.dumps(snap))
                        last_key = key
                        last_sent = now
                    await asyncio.sleep(0.12)
            except WebSocketDisconnect:
                pass
            except Exception:
                pass

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
            self.state.request_say(data.get("text", ""))
            return {"ok": True}

        @self.app.post("/api/stop")
        def _stop() -> dict:
            self.state.request_shutdown()
            return {"ok": True}

        @self.app.post("/api/mute")
        async def _mute(request: Request) -> dict:
            data = await request.json()
            self.state.muted = bool(data.get("muted", False))
            return {"ok": True}

        @self.app.post("/api/kid")
        async def _kid(request: Request) -> dict:
            data = await request.json()
            self.state.kid_mode = bool(data.get("on", True))
            return {"ok": True}

        @self.app.post("/api/volume")
        async def _volume(request: Request) -> dict:
            data = await request.json()
            v = float(data.get("volume", 2.5))
            self.state.volume = max(0.0, min(5.0, v))
            return {"ok": True}

        @self.app.post("/api/rate")
        async def _rate(request: Request) -> dict:
            data = await request.json()
            self.state.speech_rate = str(data.get("rate", "+20%"))[:10]
            return {"ok": True}

        @self.app.post("/api/energy")
        async def _energy(request: Request) -> dict:
            data = await request.json()
            self.state.energy = max(0.0, min(1.0, float(data.get("energy", 1.0))))
            return {"ok": True}

        @self.app.post("/api/gesture")
        async def _gesture(request: Request) -> dict:
            data = await request.json()
            self.state.request_gesture(data.get("name", ""))
            return {"ok": True}

        @self.app.post("/api/dance")
        async def _dance(request: Request) -> dict:
            body = {}
            try:
                body = await request.json()
            except Exception:
                pass
            self.state.request_dance(body.get("name", "macarena"))
            return {"ok": True}

        @self.app.get("/api/dances")
        def _dances() -> dict:
            from reachy_demo.dance import DANCES
            return {"dances": [
                {"key": k, "label": v["label"], "bpm": v["bpm"],
                 "duration": v["duration"]}
                for k, v in DANCES.items()
            ]}

        @self.app.get("/api/people")
        def _people() -> dict:
            from reachy_demo.memory import known_people, load_person_facts
            return {"people": [
                {"name": n, "facts": load_person_facts(n)}
                for n in known_people()
            ]}

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
