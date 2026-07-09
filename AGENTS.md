# AGENTS.md

Notes for AI coding agents working in this repo. The repo has no test suite, no
linter, no typechecker, no CI, and no `pyproject.toml` — see *Verification* below.

## What this is

Control software for a **Reachy Mini Lite** (USB variant) from Pollen Robotics.
The robot is a USB peripheral — the laptop is the computer, the Pi inside the
robot is only a USB-serial bridge for the Feetech motors. `CLAUDE.md` covers
the hardware/SDK story in full; this file adds what it misses.

## Running anything

```bash
./run.sh demos/<file>.py     # run a specific demo (foreground)
./menu.sh                    # interactive picker (7 demos)
./launch_converse.sh         # headless demo_converse: kills any orphan daemon,
                             # backgrounds the process, logs to
                             # /tmp/reachy_converse.log, prints the PID.
                             # `tail -f /tmp/reachy_converse.log` to watch.
```

Always use `run.sh` — it prepends `.venv/bin` to `PATH` and exports `PYTHONPATH`
to the repo root, both required for `reachy-mini-daemon` to be found and for
`from reachy_demo.X import …` to resolve. `FileNotFoundError:
'reachy-mini-daemon'` means you bypassed `run.sh`. `launch_converse.sh` is the
only launcher that pre-emptively `pkill`s a stale daemon before starting, so
don't run it while another demo is live.

## The actual demos (in `demos/`)

`menu.sh` is the source of truth. As of this writing:

| # | File | What it does |
|---|---|---|
| 1 | `demo_welcome.py` | Greeting + speech with layered animation |
| 2 | `demo_dance.py` | Full show with music. Swap `MUSIC = str(ROOT / "music" / "your.mp3")` |
| 3 | `demo_face_recognition.py` | Greets known faces from `faces/<name>/*.jpg`. Uses the `face_recognition` (dlib) package directly — **not** `reachy_demo.face_id` |
| 4 | `demo_tools7.py` | Parallel AI gesture picker + barge-in, any language (`AvaMultilingual` voice) |
| 5 | `demo_deepseek.py` | Like #4 but uses `opencode run` as LLM harness (DeepSeek V4 Flash via opencode). STT still via Groq. ~15 s end-to-end per turn (~8 s LLM-only); thinking ticks cover the gap |
| 6 | `demo_instant.py` | Streaming TTS — edge-tts audio streamed to the speaker as it's generated, ~0.4s time-to-first-audio |
| 7 | `demo_converse.py` | Unified: instant talk + face ID + web dashboard |

Not in the menu (superseded by `demo_converse.py`):
- `demo_dialog.py` — fluid conversation, barge-in, 700 ms turn-take, high-threshold VAD during TTS
- `demo_edge.py` — NS ambassador, online edge-tts (`AvaMultilingual` voice, pitch `+16Hz`), any language
- `demo_talk_ns.py` — NS ambassador, offline Piper voice (needs `GROQ_API_KEY` in `.env`)

> Several docs (`CLAUDE.md`, `docs/README.md`, `docs/RUN_DEMOS.md`,
> `docs/DEMOS.md`) still reference `demos/demo1_moves.py` and other dead files,
> and `docs/DEMOS.md`'s table is stale. Don't trust filenames you find in the
> docs; trust `menu.sh` and `ls demos/`.

## Shared package: `reachy_demo/`

Import these — do not reimplement in a demo:

| Module | Use it for |
|---|---|
| `daemon.py` | `start_daemon()`, `launch_daemon()`, `wait_for_daemon()`, `stop_daemon()` — manual daemon lifecycle, required because `spawn_daemon=True` is broken (see `CLAUDE.md`) |
| `animator.py` | `Animator(mini)` background thread; `set_state(Animator.IDLE/LISTENING/THINKING/SPEAKING)`; `play_gesture(name)` + `NAMED_GESTURES` dict (the LLM-driven gesture vocabulary the talking demos emit) |
| `audio.py` | `SPEAKER`, `MIC` constants; `blip`, `chirp`, `boot_beeps`, `listening_ping`, `start_thinking_ticks`, `thinking_cue`, `speaking_chime`, `error_chime`, `play_wav_blocking`, `record_utterance` (VAD via Silero), `pcm_to_wav_bytes`; mic-recovery helpers `redetect_mic`, `ensure_mic_working`, `cleanup_orphan_capture` |
| `listener.py` | **Single source of truth for the talking demos' background mic loop.** Posts `{"type":"start"|"end"|"mic_error"}` events to a queue; barge-in threshold modes; auto-recovers via `audio.redetect_mic`. Use this instead of hand-rolling VAD |
| `tts_piper.py` | `load_voice`, `synth_to_file`, `synth_and_play` (offline) |
| `tts_edge.py` | `synth_to_file`, `play_wav_blocking`, and **`stream_to_speaker(text, stop_check, on_first_audio)`** (the streaming path behind `demo_instant.py`'s low TTFA). `VOICE`=`en-US-AvaMultilingualNeural` (any language); `PITCH`=`+48Hz` gives the cute-child tone (dial back toward `+32Hz` if too chipmunky) |
| `text.py` | `SENTENCE_END` regex, `clean_for_tts` (strip markdown / roleplay emotes) |
| `groq_client.py` | `load_api_key` (reads `.env` or env var); the **multilingual STT pipeline the demos actually use**: `transcribe_lang` / `transcribe_lang_robust` → `script_language` → `language_directive`; `stream_chat`; `is_hallucination` (reject Whisper phantom text) |
| `speech_gate.py` | `is_real_speech(...)` — rms + voiced_ratio + peak_prob gate that rejects ambient noise *before* Whisper. This is the "actual noise discriminator" referenced by `audio.py` |
| `cerebras_client.py` | Optional Cerebras LLM accelerator (OpenAI-compatible endpoint). `make_client()`, `stream_chat()`, `has_key()`, `load_cerebras_key()`. Served model is **`gemma-4-31b`** — Llama-4-Scout was deprecated on Cerebras 2025-11-03 |
| `camera.py` | `CameraHub` — shared OpenCV capture thread on `/dev/video2`; `mjpeg_bytes()`, `frame_rgb()`/`frame_bgr()`, `overlay` attribute (assign `hub.overlay = drawer`; **not** a `set_overlay()` method despite the docstring) |
| `face_id.py` | `FaceIdentifier` — YuNet+SFace face ID (Apache-2.0), falls back to dlib. `identify()`, `add_person()`, **`add_person_targeted(name, frames, target_box)`** (multi-face-safe enrolment — plain `add_person` enrols the *largest* face, wrong when a bystander is bigger), `remove_person()`, `load_roster()`, `init_models()`, `mirror` flag. Used by `demo_converse.py` |
| `cues.py` | Per-language listening/thinking/"say that again" cues, synthesised once via edge-tts and cached as `cache/cue_<lang>_<kind>.wav` |
| `dance.py` | `do_macarena(mini, …)` — beat-synced Macarena at 103.4 BPM to `music/macarena.mp3`; `DANCE_KEYWORDS` multilingual trigger set. Backs `demo_dance.py` |
| `kids.py` | Kid-mode content pack (`KID_MODE_RULES`, `kid_mode_block`, `reward_line`) layered onto the base system prompt; uses `animator.NAMED_GESTURES` |
| `memory.py` | Long-term memory persisted at `memory/reachy_memory.json`. `load_memories()`, `memory_block()`, `extract_memories()`, `remember()`, `known_people()`, `load_person_facts(name)` (per-person facts under `cache/people/`). Imported by demos 4/5/6/7 |
| `live_state.py` | `LiveState` — thread-safe bridge between demo loop and web dashboard. `snapshot()`, `request_wake()`, `request_sleep()`, `request_say()`, **`request_shutdown()`** (backs the dashboard Stop button) |
| `web_server.py` | `WebDashboard` — FastAPI on :8080; MJPEG `/video`, `/status` JSON, WebSocket `/ws`, `/api/wake\|sleep\|say\|stop\|mute`, `GET /api/people` (roster from `memory.py`). Auto-reconnect frontend |
| `session_log.py` | `SessionLogger(ROOT, "demo")` — writes one numbered folder per run under `logs/<N>/` (`console.log`, `transcript.jsonl`, turn WAVs). Closest thing to reproducible debugging |
| `recorder.py` | `DiagnosticRecorder` — rolling black-box recorder (events.log + video/audio clips) capped at ~100 MB under `<base_dir>/diag/` |
| `search.py` | `web_search(query) → str`, `clean_query(text)` — DuckDuckGo helper, run in parallel with TTS |

`run.sh` exports `PYTHONPATH` to the repo root, so `from reachy_demo.X import …`
works from any demo.

## SDK constructor — must use this exact form

```python
with ReachyMini(connection_mode="localhost_only",
                media_backend="no_media",
                spawn_daemon=False) as mini:
    mini.wake_up()
    try:
        # … your moves …
    finally:
        mini.goto_sleep()        # ALWAYS — motors overheat otherwise
```

`reachy_demo.daemon.start_daemon()` handles the manual daemon launch + port
polling — used by every talking demo (including `demo_converse.py`) instead of
the broken `spawn_daemon=True`. See `CLAUDE.md` for the full boilerplate and
the motion API table (`create_head_pose`, `goto_target` vs `set_target`,
antenna/body kwargs).

## Hard-won SDK gotchas (not in CLAUDE.md)

- `set_target_antenna_joint_positions` takes a **`[left, right]` list**, not
  keyword args. Easy to misread from older docs.
- Gestures must be ≤ 2 s. Never infinite-loop `set_target`; it holds motors
  against gravity and they will overheat (there was a burning-smell incident —
  see `docs/SAFETY.md`).
- All angles in radians. `create_head_pose(..., degrees=False)` is the default.
- Safe ranges: pitch/roll ±40°, yaw ±180°, body_yaw ±160° (SDK clamps).

## Audio & camera shortcuts

- **Speaker:** always `aplay -D plughw:CARD=Audio,DEV=0 -q <wav>`. Never route
  the robot through PulseAudio sinks — routing is fragile and goes to whatever
  PipeWire picks as default. The `SPEAKER` constant in `reachy_demo.audio` is
  the right value.
- **Microphone (talking demos):** the `MIC` constant in `reachy_demo.audio`
  defaults to the robot's **`Reachy_Mini_Audio`** mic (a PipeWire source,
  captured with `pacat --record --raw --device=$MIC`), with the **laptop mic as
  fallback** only if the robot mic isn't detected. The camera's built-in mic is
  *not* used (too noisy for VAD).
- **Camera:** the SDK's `mini.media.get_frame()` needs `gst-plugins-rs`
  `webrtcsink`, which is **not installed** on this machine (~15 min `cargo
  build` to add). Use `cv2.VideoCapture('/dev/video2', cv2.CAP_V4L2)` or
  `ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video2` instead.

## File & data conventions

- `voices/en_US-amy-medium.onnx` — Piper TTS model, **gitignored** (61 MB,
  downloaded separately — see `README.md`).
- `music/` — CC-BY tracks; add your own MP3/WAV and edit the one `MUSIC = …`
  line at the top of `demo_dance.py`. Note `music/macarena.mp3` is gitignored
  specifically (copyrighted, stays local).
- `faces/<name>/*.jpg` — one subdir per known person for
  `demo_face_recognition.py`. The whole `faces/` dir is **gitignored**. Empty
  `faces/` means everyone gets the generic greeting.
- `cache/` — generated WAVs (TTS output, cue clips, per-person memory);
  gitignored, safe to delete.
- `cache/models/` — YuNet (face detect) + SFace (face ID) ONNX weights,
  **gitignored**. Auto-downloaded on first run of `demo_converse.py`.
- `memory/reachy_memory.json` — Reachy's long-term conversational memory
  (written by `reachy_demo.memory`). **Gitignored — personal data, never
  commit.**
- `logs/` — per-run session transcripts/audio from `session_log.py`
  (`logs/<N>/…`). **Gitignored.** Consumed by `tools/replay_session.py` and
  `tools/debug_one_turn.py`.
- `audio/` — gitignored scratch WAV assets (e.g. `lost_friend/`). Distinct
  from `reachy_demo/audio.py` (the module) — the name collision is a trap.
- `hello_how_are_you_many/` — tracked, standalone multi-language "hello, how
  are you" generator + interactive player. `play.py --speaker robot` routes to
  the Reachy speaker (`plughw:CARD=Audio,DEV=0`) so the robot speaks;
  `--speaker laptop|hdmi` for other sinks. Not wired into `menu.sh`.
- `.env` — contains `GROQ_API_KEY`. **Gitignored. Never commit.** Read it via
  `reachy_demo.groq_client.load_api_key(ROOT)`, which supports both `KEY:value`
  and `KEY=value` formats and falls back to the environment variable.
- `CEREBRAS_API_KEY` (optional) — if set in `.env`, `demo_converse.py` routes
  LLM calls through Cerebras (OpenAI-compatible, model `gemma-4-31b`); falls
  back to Groq otherwise. Read via `cerebras_client.load_cerebras_key(ROOT)`
  (same `KEY:value`/`KEY=` reader). No other API keys are read anywhere in the
  codebase.

## Diagnostic tools (`tools/`)

When something is wrong, start here before changing demo code:

- `tools/test_speaker.py` — verify audio output to the robot speaker
- `tools/test_mic.py` — interactive mic test across devices
- `tools/vad_test.py` — check Silero VAD on a live mic stream
- `tools/mic_test.py` — earlier mic test (kept for reference)
- `tools/replay_session.py` — replay a past run from `logs/<N>/`
  (`transcript.jsonl` + turn WAVs) to reproduce a bad conversation offline
- `tools/debug_one_turn.py` — debug a single turn from a logged session

## Orphan-daemon gotcha

If a script is killed with `kill -9`, `reachy-mini-daemon` stays running. Before
starting a new one:

```bash
pkill -9 -f "reachy-mini-daemon"
```

`reachy_demo.daemon.start_daemon()` (and `launch_converse.sh`) already do this
for you — call them instead of doing it by hand. Orphan `pacat` capture
processes are the mic equivalent; `audio.cleanup_orphan_capture()` clears them.

## Hardware preconditions

- **Back switch** must be in the **Robot / Developer** position (not
  "Computer"). In Computer mode the SDK cannot drive the motors.
- **Green LED** solid or slow-blink = Pi is up and the control stack is ready.
- Single USB-C cable exposes: `/dev/ttyACM0` (motors), `/dev/video2` (camera),
  `plughw:CARD=Audio,DEV=0` (speaker **and** robot voice mic, ALSA card 2),
  `plughw:CARD=Camera,DEV=0` (camera's built-in mic, ALSA card 1 — not used).
  If a device path shifts, `docs/HARDWARE_DIAGNOSIS.md` has the `lsusb` +
  `arecord -l` recipe to re-discover them.

## Verification

There is no test command. There is no linter, typechecker, or CI workflow. The
verification model is: run the demo and watch the robot. For logic-only edits
in `reachy_demo/`, you can `python -c "import reachy_demo.audio; …"` to check
imports and run quick syntax checks, but the only end-to-end check is on real
hardware. Do not invent a `pytest` setup unless asked.

## More reading (don't duplicate this in edits)

- `CLAUDE.md` — full SDK boilerplate, motion API table, audio/camera examples,
  HF preset libraries
- `docs/ARCHITECTURE.md` — what each USB endpoint is and who runs what
- `docs/SDK_NOTES.md` — the `spawn_daemon=True` saga and other SDK footguns
- `docs/SAFETY.md` — why `goto_sleep()` in `finally` is mandatory
- `docs/AUDIO_PIPELINE.md` — why `plughw:`, not PulseAudio
- `docs/CAMERA_PIPELINE.md` — direct UVC, no GStreamer plugin
- `docs/SETUP.md` — every apt + pip install with the reason
- `docs/HARDWARE_DIAGNOSIS.md` — `lsusb`/`arecord -l` map to re-discover device
  paths when they shift
- `docs/demo7_audit.md` — ground-up audit of `demo_tools7.py` + its 7 modules,
  with the end-to-end flow diagram and VAD thresholds; the single best doc for
  extending/debugging the talking demos
- `docs/DEMOS.md`, `docs/RUN_DEMOS.md` — per-demo details (**both stale** —
  trust `menu.sh` instead)
