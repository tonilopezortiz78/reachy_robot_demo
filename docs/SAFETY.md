# Safety — what burned, and how to avoid it

## What probably happened

During one of my long test runs, the robot was being commanded in an infinite loop (`while True` in `demo3_official_sine.py` and the upstream `examples/minimal_demo.py`). The motors were continuously active for 60+ seconds, holding the head **against gravity** at various poses.

A small servo's torque rating is fine for a quick gesture, but if you ask it to hold a position for tens of seconds, the current draw rises, the windings heat up, and the case gets hot enough to smell. The Reachy Mini uses Feetech STS3215 servos, which are rated for 15 kg·cm of torque but are not designed for continuous maximum-effort holding.

## Rules of thumb for safe movement

1. **Use `goto_target`, not raw `set_target` in a loop.** `goto_target` interpolates with a duration and then stops commanding. `set_target` keeps sending the same position, which the motor interprets as "hold here".

2. **End every demo with `mini.goto_sleep()`.** It moves the head to a relaxed pose and disables torque. If you exit without it, the last position keeps being held.

3. **Use `enable_gravity_compensation()` if you want to leave the head powered but floppy.** This makes the motors just counteract gravity — minimal current, no effort.

4. **For hold positions longer than ~2 seconds, prefer `disable_motors()` (head will fall) or `enable_gravity_compensation()`** (head stays up, no rigid hold). The user can then manually move the head.

5. **Keep gestures short.** A nod, shake, or antenna wiggle is 0.5–1 s. A 30-second continuous loop is asking for trouble.

6. **Watch the antenna servos especially** — they have the least mechanical advantage and the highest stall-current risk.

## The state machine I recommend for any demo

```python
with ReachyMini(connection_mode="localhost_only",
                media_backend="no_media",
                spawn_daemon=True) as mini:
    mini.wake_up()
    try:
        # ... do your moves here, each <2s, separated by sleeps ...
    finally:
        mini.goto_sleep()      # ALWAYS, even on exception
```

The `try/finally` is important. If anything inside the block raises, you still go to sleep.

## Emergency stop

If you smell burning or the robot is misbehaving:

```bash
pkill -9 -f reachy-mini-daemon
```

This severs the motor link. The head will fall under gravity — be ready to catch it if needed.

## Per-joint safe limits (from the SDK)

| Joint | Range |
|---|---|
| Head pitch / roll | [-40°, +40°] |
| Head yaw | [-180°, +180°] |
| Body yaw | [-160°, +160°] |
| Yaw delta (head - body) | max 65° |

The SDK clamps to these automatically. Stay well inside them.
