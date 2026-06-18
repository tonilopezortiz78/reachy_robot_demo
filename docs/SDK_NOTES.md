# SDK notes — gotchas that cost me time

## 1. `spawn_daemon=True` needs `reachy-mini-daemon` on `$PATH`

The SDK's `ReachyMini(..., spawn_daemon=True)` does:

```python
subprocess.Popen(["reachy-mini-daemon", ...])
```

The binary lives in `.venv/bin/`. If it's not on `PATH` you get:

```
FileNotFoundError: [Errno 2] No such file or directory: 'reachy-mini-daemon'
```

**Fix:** always run scripts via `./run.sh` (which prepends `.venv/bin` to `PATH`).

## 2. Default `connection_mode='auto'` tries the network first

`ReachyMini()` without arguments tries to find the robot over the network at `reachy-mini.local:8000`. On a Lite that always fails (no mDNS). The failure mode is:

```
ConnectionError: Auto connection: both localhost and remote attempts failed.
```

**Fix:** always pass `connection_mode="localhost_only"` for the Lite.

## 3. `media_backend="default"` requires GStreamer webrtcsink

The default media backend is a GStreamer pipeline that needs the `webrtcsink` Rust plugin. On a fresh laptop that plugin isn't installed, so the daemon fails to start the media server and crashes. The errors are:

```
No camera found.
Failed to create webrtcsink element. Is the GStreamer webrtc rust plugin installed?
```

**Fix:** pass `media_backend="no_media"` for now. We bypass the SDK's media pipeline entirely and talk to the camera (`/dev/video2`) and speaker (`plughw:2,0`) directly with ffmpeg/aplay.

## 4. `localhost_only=True` is deprecated

You'll see this warning:

```
DeprecationWarning: The 'localhost_only' argument is deprecated
and will be removed in a future release. Please switch to connection_mode.
```

Use `connection_mode="localhost_only"` instead.

## 5. The SDK's own TTS example needs webrtcsink

`examples/sound_tts.py` from the official repo uses `mini.media.play_sound()` which routes through the GStreamer pipeline. It will fail on the Lite without the plugin. Use a separate `aplay` (see `demos/demo2_speak.py`).

## 6. Long-running infinite-loop demos don't exit

`examples/minimal_demo.py` (and my `demo3_official_sine.py`) have a `while True:` loop that's only broken by `Ctrl+C`. If you launch them via `timeout` and don't send SIGINT, the script never exits cleanly and the daemon stays running. **Always either:**

- give them a `timeout 30 ...` wrapper, or
- press `Ctrl+C` to break out and let `ReachyMini.__exit__` clean up the daemon

## 7. `ReachyMini.__exit__` cleans up the daemon — if the script exits

Inside the `with ReachyMini(...)` block, the SDK owns the daemon. On normal exit, the daemon is shut down. On `KeyboardInterrupt` inside the `with` block, `__exit__` still runs and cleans up.

But if you `kill -9` the script, the daemon is orphaned. Always:

```bash
pkill -9 -f "reachy-mini-daemon"     # nuke any orphan daemons before starting fresh
```

## 8. Background moves vs `goto_target`

The SDK has two motion modes:

| Method | Use case |
|---|---|
| `mini.goto_target(...)` | Smooth interpolation, blocks until the move finishes, **use for everything except 10Hz+ control loops** |
| `mini.set_target(...)` | Non-blocking, instant command, **only for control loops at 10Hz+** |

`set_target` does **not** interpolate. If you call it once, the head jumps. If you call it in a 50 Hz loop, you can do real-time tracking.

## 9. `goto_sleep` is the safe end state

Always end a demo with `mini.goto_sleep()`. It moves the head to a relaxed pose and disables the motors. If you `mini.wake_up()` and then exit without `goto_sleep()`, the motors stay energised and the head will try to hold the last position — this is what caused the smell during one of my long tests. See [SAFETY.md](SAFETY.md).

## 10. `spawn_daemon=True` is broken on this machine — do not use it

Two problems discovered in SDK 1.8.3:

**Problem A — no `--no-media` flag.** The SDK spawns the daemon without `--no-media`, so the daemon tries to start the GStreamer media server, fails (missing webrtcsink plugin), and exits before the connection attempt.

**Problem B — no readiness wait.** Even if the daemon did start, `spawn_daemon=True` does `subprocess.Popen(...)` and returns immediately. `ReachyMini.__init__` then tries to connect to `ws://localhost:8000` before uvicorn is listening — connection refused.

**Fix:** start the daemon manually with `--no-media`, poll until port 8000 is up, then connect with `spawn_daemon=False`:

```python
import socket, subprocess, time

def start_daemon():
    proc = subprocess.Popen(
        ["reachy-mini-daemon", "--no-media"],
        start_new_session=True,
    )
    for _ in range(30):
        time.sleep(0.5)
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=0.3):
                return proc
        except OSError:
            pass
    raise RuntimeError("Daemon did not start within 15 s")

daemon_proc = start_daemon()
try:
    with ReachyMini(connection_mode="localhost_only",
                    media_backend="no_media",
                    spawn_daemon=False) as mini:
        ...
finally:
    daemon_proc.terminate()
    try:
        daemon_proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        daemon_proc.kill(); daemon_proc.wait()
```

## 11. Motion API keyword arguments do not exist in SDK 1.8.3

All the demos were written for an older API. In SDK 1.8.3:

**`set_target_head_pose`** takes a 4×4 numpy array, not keyword args:
```python
# WRONG (old API — raises TypeError):
mini.set_target_head_pose(pitch=0.4, yaw=0.0, degrees=False)

# RIGHT — use create_head_pose to build the matrix:
from reachy_mini.utils import create_head_pose
mini.set_target_head_pose(create_head_pose(pitch=0.4, degrees=False))
```

**`set_target_antenna_joint_positions`** takes a list, not keyword args:
```python
# WRONG (raises TypeError):
mini.set_target_antenna_joint_positions(left=0.7, right=-0.7)

# RIGHT:
mini.set_target_antenna_joint_positions([0.7, -0.7])   # [left, right]
```

**`goto_target` and `set_target`** accept the numpy array for `head=` and a plain `[left, right]` list for `antennas=`:
```python
mini.goto_target(
    head=create_head_pose(pitch=0.4, yaw=0.3, roll=0.0, degrees=False),
    antennas=[0.5, -0.5],
    duration=0.5,
    body_yaw=0.0,
)
```

`create_head_pose(pitch, yaw, roll, degrees=False)` is the single function to know. All angles default to 0. Calling it with no args returns the neutral (centre) pose.

## 12. Playing a beep on the robot speaker before a demo

`ffmpeg`'s `lavfi` source can generate a sine wave directly to ALSA with no temp file:

```python
import subprocess

def beep(freq=880, duration=0.3):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration}",
         "-f", "alsa", "plughw:2,0"],
        check=False,
    )
```

Use `plughw:2,0` (direct ALSA to the robot's USB speaker), not a PulseAudio sink name. This is consistent with the rest of the audio pipeline — see [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md).
