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
./run.sh demos/<file>.py     # run a specific demo
./menu.sh                    # interactive picker (8 demos)
```

Always use `run.sh` — it prepends `.venv/bin` to `PATH`, which is required for
`reachy-mini-daemon` to be found. `FileNotFoundError: 'reachy-mini-daemon'`
means you bypassed `run.sh`.

## The actual demos (in `demos/`)

`menu.sh` is the source of truth. As of this writing:

| # | File | What it does |
|---|---|---|
| 1 | `demo_welcome.py` | Greeting + speech with layered animation |
| 2 | `demo_dance.py` | Full show with music. Swap `MUSIC = str(ROOT / "music" / "your.mp3")` |
| 3 | `demo_talk_ns.py` | NS ambassador, offline Piper voice (needs `GROQ_API_KEY` in `.env`) |
| 4 | `demo_face_recognition.py` | Greets known faces from `faces/<name>/*.jpg` |
| 5 | `demo_edge.py` | NS ambassador, online edge-tts (`AvaMultilingual` voice, pitch `+16Hz` for a cute tone), any language |
| 6 | `demo_dialog.py` | Fluid conversation — barge-in, 700 ms turn-take, high-threshold VAD during TTS |
| 7 | `demo_tools7.py` | Parallel AI gesture picker + barge-in, any language (`AvaMultilingual` voice) |
| 8 | `demo_deepseek.py` | Like #7 but uses `opencode run` as LLM harness (DeepSeek V4 Flash via opencode). STT still via Groq. ~8s LLM latency (opencode overhead) — thinking ticks cover the gap. |

> Several docs (`CLAUDE.md`, `docs/README.md`, `docs/RUN_DEMOS.md`) still
> reference `demos/demo1_moves.py` — that file no longer exists. Don't trust
> filenames you find in the docs; trust `menu.sh` and `ls demos/`.

## Shared package: `reachy_demo/`

Import these — do not reimplement in a demo:

| Module | Use it for |
|---|---|
| `daemon.py` | `start_daemon()`, `launch_daemon()`, `wait_for_daemon()`, `stop_daemon()` — manual daemon lifecycle, required because `spawn_daemon=True` is broken (see `CLAUDE.md`) |
| `animator.py` | `Animator(mini)` background thread; `set_state(Animator.IDLE/LISTENING/THINKING/SPEAKING)` |
| `audio.py` | `SPEAKER`, `MIC` constants; `blip`, `chirp`, `boot_beeps`, `listening_ping`, `your_turn_chime`, `thinking_blips`, `speaking_chime`, `error_chime`, `play_wav_blocking`, `record_utterance` (VAD via Silero), `pcm_to_wav_bytes` |
| `tts_piper.py` | `load_voice`, `synth_to_file`, `synth_and_play` (offline) |
| `tts_edge.py` | `synth_to_file`, `play_wav_blocking`; single `VOICE` constant (`en-US-AvaMultilingualNeural`, any language) plus `RATE`/`PITCH`/`VOL` tuning — `+16Hz` pitch gives the cute tone (online, needs internet) |
| `text.py` | `SENTENCE_END` regex, `clean_for_tts` (strip markdown / roleplay emotes) |
| `groq_client.py` | `load_api_key` (reads `.env` or env var), `transcribe`, `stream_chat` |

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
polling. See `CLAUDE.md` for the full boilerplate and the motion API table
(`create_head_pose`, `goto_target` vs `set_target`, antenna/body kwargs).

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
- **Microphone (talking demos):** `pacat --record --raw --device=<PipeWire
  source> ...` — the `MIC` constant points at the laptop mic, not the camera's
  built-in mic (camera mic is too noisy for VAD).
- **Camera:** the SDK's `mini.media.get_frame()` needs `gst-plugins-rs`
  `webrtcsink`, which is **not installed** on this machine (~15 min `cargo
  build` to add). Use `cv2.VideoCapture('/dev/video2', cv2.CAP_V4L2)` or
  `ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video2` instead.

## File & data conventions

- `voices/en_US-amy-medium.onnx` — Piper TTS model, **gitignored** (61 MB,
  downloaded separately — see `README.md`).
- `music/` — CC-BY tracks; add your own MP3/WAV and edit the one `MUSIC = …`
  line at the top of `demo_dance.py`.
- `faces/<name>/*.jpg` — one subdir per known person for
  `demo_face_recognition.py`. Empty `faces/` means everyone gets the generic
  greeting.
- `cache/` — generated WAVs (TTS output, etc.); gitignored, safe to delete.
- `.env` — contains `GROQ_API_KEY`. **Gitignored. Never commit.** Read it via
  `reachy_demo.groq_client.load_api_key(ROOT)`, which supports both `KEY:value`
  and `KEY=value` formats and falls back to the environment variable.

## Diagnostic tools (`tools/`)

When something is wrong, start here before changing demo code:

- `tools/test_speaker.py` — verify audio output to the robot speaker
- `tools/test_mic.py` — interactive mic test across devices
- `tools/vad_test.py` — check Silero VAD on a live mic stream
- `tools/mic_test.py` — earlier mic test (kept for reference)

## Orphan-daemon gotcha

If a script is killed with `kill -9`, `reachy-mini-daemon` stays running. Before
starting a new one:

```bash
pkill -9 -f "reachy-mini-daemon"
```

`reachy_demo.daemon.start_daemon()` already does this for you — call it
instead of doing it by hand.

## Hardware preconditions

- **Back switch** must be in the **Robot / Developer** position (not
  "Computer"). In Computer mode the SDK cannot drive the motors.
- **Green LED** solid or slow-blink = Pi is up and the control stack is ready.
- Single USB-C cable exposes: `/dev/ttyACM0` (motors), `/dev/video2` (camera),
  `plughw:CARD=Audio,DEV=0` (speaker, ALSA card 2), `plughw:CARD=Camera,DEV=0`
  (camera's built-in mic, ALSA card 1 — not used by the current demos).

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
- `docs/DEMOS.md`, `docs/RUN_DEMOS.md` — per-demo details (note: the file
  list in `RUN_DEMOS.md` is stale; trust `menu.sh` instead)
