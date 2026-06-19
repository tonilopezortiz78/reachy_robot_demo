"""
reachy_demo/daemon.py — Start and stop the reachy-mini-daemon process.
"""
import socket
import subprocess
import time


def start_daemon() -> subprocess.Popen:
    """Kill any existing daemon, start a fresh one, and wait until port 8000 is up."""
    subprocess.run(["pkill", "-9", "-f", "reachy-mini-daemon"], check=False)
    time.sleep(0.3)
    proc = subprocess.Popen(
        ["reachy-mini-daemon", "--no-media"], start_new_session=True,
    )
    for _ in range(30):
        time.sleep(0.5)
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=0.3):
                return proc
        except OSError:
            pass
    raise RuntimeError("Daemon did not start within 15 s")


def stop_daemon(proc: subprocess.Popen) -> None:
    """Terminate the daemon process, killing it if it doesn't exit within 8 s."""
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
