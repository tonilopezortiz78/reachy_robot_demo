# demo_tools7 — full audit

A ground-up walkthrough of `demos/demo_tools7.py` and the seven `reachy_demo`
modules it touches. Useful when extending the demo, debugging a bad session,
or rewriting the dialog engine from scratch.

The demo is an **NS-ambassador robot** that listens continuously in any
language, streams an LLM reply, picks its own opening gesture in parallel,
and supports barge-in. Single demo file (`demos/demo_tools7.py`, 740 lines)
plus the `reachy_demo/` shared package.

## Run

```bash
./run.sh demos/demo_tools7.py   # sets PATH for reachy-mini-daemon
```

Menu entry **7**. Hardware preconditions: back switch in Robot/Developer
position, green LED up, USB-C connected. See `CLAUDE.md` for the full list.

---

## 1. End-to-end flow

```
                         ┌──────────────────────────────────────────────┐
                         │              demos/demo_tools7.py           │
                         │                                              │
   mic (pacat)──────────▶│ ContinuousListener (thread)                 │
                         │   Silero VAD 16 kHz, threshold 0.45/0.75    │
                         │   posts events {"start"}, {"end", pcm}      │
                         │           │                                  │
                         │           ▼                                  │
                         │   main loop (events.get)                     │
                         │     ├─ "start"   → ignore (filter chatter)  │
                         │     └─ "end"     → THINKING + speak_cue     │
                         │                  + thinking-tick loop        │
                         │                  + STT (Groq Whisper)        │
                         │                  + LLM (Groq, streaming)     │
                         │                  + TTS (edge-tts)            │
                         │                  + gesture picker (parallel)│
                         │                  → back to LISTENING         │
                         │                                              │
                         │ DialogEngine.speak()                         │
                         │   - one-ahead TTS synthesis                  │
                         │   - barge-in at any point                    │
                         │   - mid-stream gesture markers               │
                         └──────────────────────────────────────────────┘
                                       │              │            │
                       ┌───────────────┘              │            └──────────────┐
                       ▼                              ▼                           ▼
   reachy_demo/audio.py           reachy_demo/animator.py        reachy_demo/tts_edge.py
   - blip / chirp (ffmpeg tones)  - state machine (IDLE /        - edge-tts in background
   - listening_ping                 LISTENING / THINKING /        asyncio loop, +48 Hz
   - thinking_cue / ready_cue       SPEAKING)                    AvaMultilingual voice
   - start_thinking_ticks (loop)   - base sine wave +            - resamples to 48 kHz WAV
   - play_wav_blocking              aliveness micro-gestures    - aplay on the speaker
   - SPEAKER = plughw:CARD=Audio  - play_gesture (HF library)
```

State machine (the spine of the demo):

```
   ┌──── start ────▶ IDLE ──wake_up()──┐
   │                                   ▼
   │                              SPEAKING ─speak_greeting()─▶ LISTENING
   │                                   ▲                          │
   │                                   │                          ▼
   │                                   └──── play aplay reply ──── end-of-utterance
   │                                   │                          │
   │                                   │                          ▼ THINKING
   │                                   │                          │
   │                                   └── start_thinking_ticks   │
   │                                       stop.set()  ◀──────────┘
   │                                       (right before 1st aplay)
   ▼ on Ctrl-C                            │
   finally: mini.goto_sleep()             │
                                           ▼ LISTENING
```

---

## 2. The seven shared modules

### `reachy_demo/daemon.py` (45 lines)

| Function | Notes |
|---|---|
| `launch_daemon()` | `pkill -9 -f reachy-mini-daemon`, then `Popen(['reachy-mini-daemon', '--no-media'], start_new_session=True)`. Returns immediately — daemon is starting. |
| `wait_for_daemon(proc, timeout=20)` | Polls `127.0.0.1:8000` every 100 ms. |
| `start_daemon()` | `wait_for_daemon(launch_daemon())` — convenience. |
| `stop_daemon(proc)` | `terminate` → wait 8 s → `kill`. |

Demo 7 calls `launch_daemon()` + `wait_for_daemon()` separately so it can log
the gap.

### `reachy_demo/audio.py` (305 lines)

Hardware constants: `SPEAKER = "plughw:CARD=Audio,DEV=0"` (fixed string;
always used as `aplay -D plughw:CARD=Audio,DEV=0`).
`MIC` is auto-detected at import via `pactl list short sources`, preferring
`Reachy_Mini_Audio` over `Reachy_Mini_Camera` over the laptop fallback.

Tones are pure ffmpeg: `blip` = damped sine, `chirp` = linear-frequency
sweep. Non-blocking variants use `Popen`.

**Demo-7 callers:** `boot_beeps` (startup), `speaking_chime` (pre-greeting),
`start_thinking_ticks` (during STT/LLM wait), `error_chime` (catch-all),
`pcm_to_wav_bytes` (wraps raw PCM for Whisper).

**`start_thinking_ticks(stop_event)`** is a background thread that loops
a slow sci-fi "thinking scan" pattern — rising chirp (380→1100 Hz over
0.55 s), gap 1.3 s, second rising chirp (460→1300 Hz over 0.40 s), gap
1.5 s, then repeat. Sleeps are broken into 50 ms slices so the event
takes effect within ~50 ms (not the 1.5 s gap duration). The thread
is `daemon=True`; demo 7 also `.join(timeout=1.0)`s it in its
`finally` for clean shutdown. Uses the non-blocking `chirp_nb()`
helper to keep ffmpeg from blocking the loop.

### `reachy_demo/animator.py` (367 lines)

A background thread at 20 Hz reads `self.state` under a lock and sends
`mini.set_target(head, antennas=[l,r], body_yaw)` with sine-wave
parameters. State-dependent values for pitch/yaw/roll/body_yaw/antennas
are in `_loop`. Aliveness layer (`_AlivenessLayer`) superimposes random
micro-gestures from `GESTURE_TEMPLATES` (head tilts, antenna flicks,
gaze shifts) on a Poisson process and an independent per-antenna
random walk. `play_gesture(name)` is fire-and-forget HF library
moves — it sets `_gesture_active` and the loop suspends base +
aliveness for the duration so the gesture reads cleanly.

**Demo-7 states used:** `SPEAKING` (greeting, reply), `THINKING`
(after utterance end), `LISTENING` (default). `IDLE` is the implicit
rest state but the demo never sets it.

### `reachy_demo/cues.py` (107 lines)

Spoken turn-taking cues ("I'm listening!" / "Let me think…") in the
visitor's language. Pre-generated on first use, cached in
`cache/cue_<lang>_<kind>.wav`. `speak_cue(listener, kind, lang)` is
the safe call site: it `mute()`s the listener first, plays the cue
via `aplay`, waits, then `unmute()`s. The mute is mandatory because
the speaker and mic are the same USB device on this robot — without
muting, the robot hears its own cue and starts a phantom conversation.
Languages are listed in `CUE_PHRASES`; anything not listed falls back
to English. `prewarm(lang)` synthesises both cues in a background
thread at startup so the first use has zero latency.

### `reachy_demo/groq_client.py` (174 lines)

`load_api_key(root)` reads `GROQ_API_KEY` from `.env` (supports both
`KEY:value` and `KEY=value` formats) or the environment. The two STT
entry points are `transcribe` (fixed language) and `transcribe_lang`
(verbose_json — returns text + detected language).

**`resolve_language(text, whisper_lang)`** is the key to multilingual
correctness: it first tries `script_language(text)` (Unicode ranges —
kana ⇒ Japanese, hangul ⇒ Korean, CJK ideographs ⇒ Chinese, etc.) and
falls back to Whisper's audio label only if no script match. This is
the fix for "asked in Japanese, replied in Spanish" — Whisper's audio
detector is unreliable on short clips, but the kana in the transcript
is unambiguous. `language_directive(lang)` then builds a recency-biased
system instruction ("Write your ENTIRE reply in Japanese. Do not switch
to any other language.") that is appended AFTER history so it dominates
the LLM's language choice.

`stream_chat` is defined but **not used by demo 7** — demo 7 calls
`client.chat.completions.create(stream=True)` directly so it can also
fire `pick_action` in the same window.

### `reachy_demo/text.py` (36 lines)

`SENTENCE_END` is the regex `(?<=[.!?])\s+` — used in the streaming
LLM loop to split the accumulating buffer into speakable sentences.
`clean_for_tts` strips roleplay emotes (`*beep*`), markdown
(`**bold**`, `[text](url)`), bracketed emotion tags in any language
(`[高兴]`, `[laughs]`, `【中文】`), and bullet markers. Critical for
TTS — without this the robot would literally say "asterisk beep
asterisk".

### `reachy_demo/tts_edge.py` (84 lines)

A single `asyncio` event loop runs in a daemon thread for the process
lifetime. Every `synth_to_file(text)` call submits a coroutine via
`run_coroutine_threadsafe`, which reuses the TLS connection. Output
is resampled to 48 kHz and aplay'd. Constants:

- `VOICE = "en-US-AvaMultilingualNeural"` (same voice for all langs)
- `RATE = "+30%"` (snappier than default)
- `PITCH = "+48Hz"` (cute childlike — never set to 0)
- `VOL = "2.5"` (+8 dB, louder than unity)

### `reachy_demo/session_log.py` (84 lines)

`SessionLogger(root, demo_name)` creates `data/<N>/` (auto-incremented)
with `console.log`, `transcript.jsonl`, `audio/turn_NNN.wav`. Every
turn is logged with: `stt` (audio path + whisper language + script
override + final language + directive + transcript + timings), `llm_request`
(the full Groq payload), `llm_reply` (the segments), `spoken`,
`interrupted`, `error`. Created FIRST in `main()` so even a startup
crash is captured.

---

## 3. The dialog engine — the part that matters

### Three concurrent things

When the user finishes speaking, the main loop fires:

1. **Thinking tick** — soft blip-blip loop until the first TTS segment
   is about to play. Added in the same change that wired the listening
   pose. Pattern: `(tick 900 Hz 50 ms), (gap 400 ms), (tick 1100 Hz
   50 ms), (gap 800 ms)`. Kills within 50 ms of `stop_thinking.set()`.
2. **STT (Whisper large-v3, verbose_json)** — ~400 ms typical. Returns
   `(text, whisper_lang)`. The script-override then decides the
   actual `final_lang`.
3. **LLM streaming** — two parallel Groq calls fire at the same moment:
   - `pick_action()` — non-streaming, `max_tokens=5`, returns one
     word from the gesture vocabulary (or `none` ~75% of the time).
     Completes in ~150 ms and fires the gesture *before* the first
     spoken word.
   - `chat.completions.create(stream=True)` — the actual reply. As
     soon as a sentence boundary is detected in the stream, the
     first segment is submitted to TTS. N+1 is queued while N plays.

### Streaming TTS (one-ahead)

```
   stream chunk arrives
     ↓
   buffer += chunk.delta.content
     ↓
   SENTENCE_END.split(buffer)  ──▶ parts
     ↓
   for each complete part:
     seg = _extract_segment(part)   # pull [gesture] marker, clean_for_tts
     if seg:
       segments.append(seg)
       if next_future is None:
         next_future = tts_pool.submit(synth_to_file, seg.text)
   ↓
   for i, seg in enumerate(segments):
     wav = next_future.result()      # wait for current
     next_future = tts_pool.submit(synth_to_file, segs[i+1].text)  # queue next
     stop_thinking.set()             # kill ticks (first iter only)
     self.anim.set_state(SPEAKING)
     self._tts_proc = Popen(['aplay', '-D', SPEAKER, '-q', wav])
     while self._tts_proc.poll() is None:
       if self._drain_barge_in(timeout=0.08):
         return None
```

### Barge-in

`set_threshold_mode("barge_in")` raises the VAD threshold to 0.75
(combined with 6 consecutive trigger frames ≈ 200 ms). When
`_drain_barge_in` sees a `start` event it kills the TTS subprocess
(terminate → 0.4 s → kill) and returns `None` from `speak()`. The
outer loop sees the `None` reply, logs the interruption, and returns
to LISTENING.

The cue helpers (cues.py) **already** mute the listener around their
own playback — that's why `speak_cue("thinking", ...)` doesn't
self-trigger even though the speaker and mic are the same device.

### Romaji-misdetection retry

Whisper sometimes hears Japanese audio and returns
`language="indonesian", text="Konnichiwa"` — a transliteration that
defeats our script-based language override (kana ⇒ Japanese). To
handle this, `transcribe_lang_robust()` in `groq_client.py`:

1. Calls `transcribe_lang()` normally.
2. If `script_language(text) is None` AND `whisper_lang` is in a
   small set of known ambiguous languages (`indonesian`, `malay`,
   `vietnamese`, `filipino`, `tagalog`, `turkish`), retries once
   with `language="ja"` (forced).
3. If the retry returns text containing kana, returns that — the
   script-override then identifies it as Japanese and the LLM
   directive is built correctly.
4. Otherwise keeps the original transcription.

Costs one extra Whisper call only on misdetection (the common case
is fast). The whole interaction for a real Japanese utterance goes
~700ms → ~1.4s, which is fine.

### Inline gesture markers

The LLM may emit `[acknowledge]`, `[yes]`, `[thinking]` etc. at the
start of any sentence. `_extract_segment` pulls the first one and
`clean_for_tts` removes any later ones. The first marker is played
*only if* `pick_action` didn't already fire a real gesture — stacking
two big movements at the start of a turn looks violent.

---

## 4. Bug list (and what was fixed)

| # | Sev | File:line | Issue | Fixed? |
|---|---|---|---|---|
| 1 | bug | `reachy_demo/audio.py:158-159` | The thinking-tick `pattern` used `(f, dur, vol)` tuples with `0.40` and `0.80` in the **frequency** slot for pause rows. Result: `f=0.40` ⇒ near-DC silence blip, and `pause = dur if dur is not None else 0.0` always returned `0.0` for those rows, so the 400/800 ms gaps were zero. Replaced with `(kind, f_or_pause, dur, vol)` 4-tuples; pause rows have `kind="gap"` and `dur` is the actual gap. | yes |
| 2 | bug | `demos/demo_tools7.py:650-651` | `stop_thinking` was a per-iteration local variable, so the outer `finally` (Ctrl-C path) couldn't kill the tick thread. Hoisted to the `try` scope, added `tick_thread` ref, and `join(timeout=1.0)` in the finally. | yes |
| 3 | bug | `demos/demo_tools7.py:466-494` (playback loop) | The "1-ahead" TTS pipeline (synth N+1 while playing N) raced with the exclusive `plughw:CARD=Audio,DEV=0` speaker: if segment 1 was short and its `aplay` was still finishing when segment 2's `aplay` launched, the second one failed with "Device or resource busy" and was silently dropped. Replaced with fully-serial TTS: submit next future only AFTER the current aplay exits. | yes |
| 4 | bug | `demos/demo_tools7.py` + `reachy_demo/groq_client.py` | Whisper misdetected Japanese audio as Indonesian and transliterated the text into romaji ("Konnichiwa" instead of "こんにちは"), defeating the script-override defence. Added `transcribe_lang_robust()` in `groq_client.py` that retries once with `language="ja"` when the first pass returned romaji + an ambiguous Whisper label. The script-override then catches the kana on the second pass. | yes |
| 5 | note | `demos/demo_tools7.py:619-626` | The opening greeting calls `play_gesture("greeting")` (fire-and-forget, ~2 s) and then immediately `speak_greeting(...)` (blocking, ~2 s). The base animation is suspended for the duration of the gesture, so the head is held by the HF preset while the greeting is spoken. This is actually a fine "act then speak" feel — the demo's own description matches this. Documented, not changed. | n/a |
| 6 | note | `demos/demo_tools7.py:629-630` | `listener.start()` then `speak_cue(listener, ...)` — the cue does `listener.mute()` before play, but pacat is already streaming. The first frame or two of audio arrives unmated. In practice the speaker→mic bleed is too short to trigger VAD at 0.45. Not worth fixing. | n/a |
| 7 | note | `demos/demo_tools7.py:613` | `ThreadPoolExecutor(max_workers=2)` for `pick_action` — only one task is ever submitted per turn. `max_workers=1` would suffice. Harmless. | n/a |
| 8 | note | `reachy_demo/groq_client.py:154-173` | `stream_chat` is dead code (demo 7 streams directly). Kept for other demos (`demo_edge.py` and `demo_dialog.py` import it). | n/a |

---

## 5. Operational notes

### `pacat` startup vs first `aplay`

The robot speaker (`plughw:CARD=Audio,DEV=0`) is exclusive. If the
`boot_beeps` ffmpeg is still writing when the first `speak_cue` calls
`aplay`, the second process will fail with "Device or resource busy"
and the cue will be silently dropped. Demo 7 already has a `time.sleep(0.15)`
between `boot_beeps` and the next speaker use, which is enough.

### `pacmd` / `pactl` mic

`MIC` is detected at *import* time. If the robot is plugged in *after*
the demo starts, `MIC` is the laptop fallback and VAD will see room
noise. Restart the demo after plugging in the USB.

### Orphan daemon

If the demo is killed with `kill -9`, the daemon stays running. The
next `launch_daemon()` calls `pkill -9 -f reachy-mini-daemon` first,
so this is safe — but you will see "Daemon ready 0.1s" because the
old daemon is still listening on port 8000 when the new one starts.
If the new daemon's port fails to bind, `wait_for_daemon` will time
out at 20 s. Symptom: the demo hangs at startup. Fix: manually
`pkill -9 -f reachy-mini-daemon` and retry.

### What the data dir contains

`data/<N>/` per run:
- `console.log` — human timeline, `event()` calls + echoed stdout
- `transcript.jsonl` — one JSON object per turn, see `SessionLogger.turn`
- `audio/turn_NNN.wav` — the exact PCM Whisper heard (replayable)

Useful for debugging a bad session: replay `turn_007.wav` to
`aplay -D plughw:CARD=Audio,DEV=0` to hear what the user said, and
read the matching `llm_request` entry to see the full prompt.

### What's NOT in the demo

- No camera/VLM (intentional — `media_backend="no_media"`)
- No LLM gesture markers in the system prompt for non-gesture markers
  like `[smile]` — the regex is a fixed allow-list
- No memory across sessions — `history = []` at start
- No user identification — every visitor is "the user"

---

## 6. Extending the demo

Most likely extension points:

| Want to | Edit |
|---|---|
| Add a new gesture | Add `(name, hf_preset)` to `NAMED_GESTURES` in `animator.py` and add the name to the LLM's system prompt marker list. |
| Add a new language | Add `(listening, thinking)` to `CUE_PHRASES` in `cues.py`. `language_directive` already accepts any string. |
| Add a new state | Add it to `Animator.STATES` + a `state == self.X` branch in `_loop`. Then add a `GESTURE_RATE`, `ANTENNA_NEUTRAL`, `ANTENNA_LIVENESS` entry. |
| Change the voice | Edit `VOICE`, `RATE`, `PITCH`, `VOL` in `tts_edge.py`. |
| Change turn-take | `SILENCE_END_MS` in `audio.py` (only used by `record_utterance`; the streaming listener uses `SILENCE_MS` in demo 7 itself, line 67). |
| Add inline gesture | Add a new `[name]` token to the LLM's allowed list; `GESTURE_MARKER` regex is built dynamically from `NAMED_GESTURES`. |

---

## 7. Cross-references

- `CLAUDE.md` — full SDK boilerplate, motion API, audio examples
- `docs/SAFETY.md` — why `goto_sleep()` in `finally` is mandatory
- `docs/AUDIO_PIPELINE.md` — why `plughw:`, not PulseAudio sinks
- `docs/SDK_NOTES.md` — `spawn_daemon=True` saga and other SDK footguns
- `AGENTS.md` — repo conventions and verification model
