# Running the demos

## Quick start

```bash
cd /home/tony/software/robots/reachy
./menu.sh
```

`menu.sh` lists the eight working demos and runs the one you pick. Each one:

- starts the SDK daemon (via manual `reachy-mini-daemon` launch with `spawn_daemon=False`)
- wakes the robot
- does its thing
- goes to sleep
- shuts the daemon down

If something hangs, `Ctrl+C` will trigger `goto_sleep` and stop the daemon. As a nuclear option:

```bash
pkill -9 -f reachy-mini-daemon
```

## The eight working demos

| # | File | What it does | Expected runtime |
|---|---|---|---|
| 1 | `demo_welcome.py` | Greeting speech + attentive listening pose | ~25 s |
| 2 | `demo_dance.py` | Full Macarena performance with beat-synced music | ~30 s |
| 3 | `demo_face_recognition.py` | Loads known faces from `faces/<name>/`, greets by name, tracks faces in real time | ~90 s (cooldown between greetings) |
| 4 | `demo_tools7.py` | Conversational LLM with barge-in and parallel gesture picker, fast (~1 s turn-taking) | varies |
| 5 | `demo_deepseek.py` | Same as demo 4 but uses DeepSeek V4 Flash (deeper thinking, ~15 s latency) | varies |
| 6 | `demo_instant.py` | Streaming TTS — starts talking ~0.4 s after LLM produces a sentence | varies |
| 7 | `demo_converse.py` | Instant talk + face ID onboarding + web dashboard at `http://localhost:8080` | varies |
| 8 | `demo_hackathon.py` | Dual-view tabbed dashboard (`/#stage` projector view + `/#control` operator), kid-friendly | varies |

## The official examples (also in `demos/`)

These are untouched copies of upstream Pollen examples. They are useful as reference, but on the Lite **without the webrtcsink plugin** they will fail at the `ReachyMini(...)` line because they don't pass `media_backend="no_media"`. Don't run them directly — look at the code, copy the parts you want into a new script, and add `media_backend="no_media", connection_mode="localhost_only", spawn_daemon=True` to the constructor.

Files:

- `official_minimal.py` — simplest connection example
- `official_sequence.py` — choreographed multi-step sequence
- `official_recorded_moves.py` — plays moves from a HF dataset
- `official_sound_play.py` — plays a sound from the robot's built-in library
- `official_sound_tts.py` — uses the SDK's own TTS pipeline
- `official_take_picture.py` — uses `mini.media.get_frame()`
- `official_look_at_image.py` — face detection → look-at target
- `official_joy_controller.py` — gamepad control
- `official_sound_record.py` — mic capture
- `official_sound_doa.py` — direction-of-arrival mic processing
- `official_rerun_viewer.py` — visualises the joint state in Rerun
- `official_mini_head_position_gui.py` — slider GUI
- `official_imu_example.py` — IMU (Lite has no IMU; will return zeros)
- `official_custom_media_manager.py` — shows how to plug a custom media backend

## Why my demos use `connection_mode="localhost_only", spawn_daemon=False, media_backend="no_media"`

The defaults assume a Wireless robot on the LAN. The Lite is none of those things — see [SDK_NOTES.md](SDK_NOTES.md). The `spawn_daemon=False` pattern requires manual daemon startup with `--no-media` flag and a poll loop; see the pattern in [SDK_NOTES.md](SDK_NOTES.md) section 10.
