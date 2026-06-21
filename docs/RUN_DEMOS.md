# Running the demos

## Quick start

```bash
cd /home/tony/software/robots/reachy
./menu.sh
```

`menu.sh` lists the five working demos and runs the one you pick. Each one:

- starts the SDK daemon (via `spawn_daemon=True`)
- wakes the robot
- does its thing
- goes to sleep
- shuts the daemon down

If something hangs, `Ctrl+C` will trigger `goto_sleep` and stop the daemon. As a nuclear option:

```bash
pkill -9 -f reachy-mini-daemon
```

## The five working demos

| # | File | What it does | Expected runtime |
|---|---|---|---|
| 1 | `demo1_moves.py` | Nods, shakes, tilts, antenna wiggle | ~12 s |
| 2 | `demo2_speak.py` | Speaks "Hello! Welcome to Network School" via Piper TTS, with head + antenna animation | ~10 s (depends on speech length) |
| 3 | `demo3_official_sine.py` | Smooth sine-wave sway (press Ctrl+C to stop) | infinite (or until killed) |
| 4 | `demo4_official_moves.py` | Plays moves from Pollen's HF dataset. `-d emotions` (default) or `-d dances`. `--full` to play all | ~1 min for 3 moves, ~10 min for all |
| 5 | `demo5_camera.py` | Snapshots the camera and scans head left/right, capturing frames at each pose | ~8 s |

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

## Why my demos use `connection_mode="localhost_only", spawn_daemon=True, media_backend="no_media"`

The defaults assume a Wireless robot on the LAN. The Lite is none of those things — see [SDK_NOTES.md](SDK_NOTES.md).
