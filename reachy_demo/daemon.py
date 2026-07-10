"""
reachy_demo/daemon.py — Start and stop the reachy-mini-daemon process.
"""
import os
import signal
import socket
import subprocess
import time


def launch_daemon() -> subprocess.Popen:
    """Kill any existing daemon and start a fresh one. Returns immediately — not ready yet."""
    subprocess.run(["pkill", "-9", "-f", "reachy-mini-daemon"], check=False)
    time.sleep(0.3)
    return subprocess.Popen(
        ["reachy-mini-daemon", "--no-media"], start_new_session=True,
    )


def wait_for_daemon(proc: subprocess.Popen, timeout: float = 20.0) -> subprocess.Popen:
    """Block until port 8000 is listening (or timeout)."""
    t0 = time.time()
    deadline = t0 + timeout
    while time.time() < deadline:
        time.sleep(0.1)
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=0.3):
                print(f"  Daemon ready  {time.time()-t0:.1f}s", flush=True)
                return proc
        except OSError:
            pass
    raise RuntimeError(f"Daemon did not start within {timeout:.0f} s")


def start_daemon() -> subprocess.Popen:
    """Kill any existing daemon, start a fresh one, and wait until port 8000 is up."""
    return wait_for_daemon(launch_daemon())


def reconnect_reachy_mini(old_mini, daemon_proc, log=None, max_attempts=5):
    """Recover from a dead SDK connection (USB link dropped — e.g. a vigorous dance
    jostled the cable). Restarts the daemon only if port 8000 isn't answering (which
    also re-discovers /dev/ttyACM* if it re-enumerated), then rebuilds the ReachyMini
    client and wakes it. Returns (new_mini, daemon_proc). Raises after max_attempts so
    the caller's supervisor can fall back to a full process restart.
    """
    from reachy_mini import ReachyMini
    try:
        old_mini.client.disconnect()
    except Exception:
        pass
    for attempt in range(1, max_attempts + 1):
        try:
            daemon_up = False
            try:
                with socket.create_connection(("127.0.0.1", 8000), timeout=0.5):
                    daemon_up = True
            except OSError:
                daemon_up = False
            if not daemon_up:
                if log:
                    log.event(f"  [reconnect] daemon down — restarting "
                              f"(attempt {attempt}/{max_attempts})")
                daemon_proc = start_daemon()
            new_mini = ReachyMini(connection_mode="localhost_only",
                                  media_backend="no_media", spawn_daemon=False)
            new_mini.wake_up()
            if log:
                log.event(f"  [reconnect] robot reconnected (attempt {attempt})")
            return new_mini, daemon_proc
        except Exception as e:
            if log:
                log.error("reconnect", e)
            time.sleep(min(2 * attempt, 8))   # backoff — avoids a tight restart loop
    raise RuntimeError(f"Could not reconnect to Reachy after {max_attempts} attempts")


def stop_daemon(proc: subprocess.Popen) -> None:
    """Terminate the daemon process, killing it if it doesn't exit within 8 s.

    The daemon is launched with start_new_session=True, so it owns a whole
    process group — signal the group, not just the parent, or its children
    keep running (and holding the motors/port). Falls back to plain
    terminate/kill on the parent if group signalling isn't possible.
    """
    def _signal_group(sig) -> bool:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            return True
        except ProcessLookupError:
            return True   # group already gone — nothing left to signal
        except OSError:
            return False  # fall back to signalling the parent only

    if not _signal_group(signal.SIGTERM):
        proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        if not _signal_group(signal.SIGKILL):
            proc.kill()
        proc.wait()
