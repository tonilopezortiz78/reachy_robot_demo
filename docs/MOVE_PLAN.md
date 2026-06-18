# Move plan — one small move, awaiting your approval

**Before I run anything else, please confirm.** I will not run a movement script until you say go.

## What I want to do

A single, **very small** head nod. The plan, in order:

| Step | What | Duration | Why |
|---|---|---|---|
| 1 | `mini.wake_up()` | ~0.5 s | Energise motors |
| 2 | `mini.goto_target(head=pitch -10°, duration=0.6 s)` | 0.6 s | Tilt head down a small amount |
| 3 | `mini.goto_target(head=pitch +0°, duration=0.6 s)` | 0.6 s | Return to neutral |
| 4 | `mini.goto_sleep()` | ~0.8 s | Disable torque, robot goes floppy-safe |

**Total motor activity: about 2.5 seconds.** No loops. No holds. No antenna movement. Then sleep.

## Why this should not smell like burning

- Each command lasts 0.6 s, then the motor is idle (not holding) for 0.6 s
- The angle change is 10° — well within the safe range
- We sleep immediately afterwards, releasing all torque
- No continuous loop, no held position

## What you should see

- The head tips down ~10° (just a small nod), then returns to centre
- Head becomes floppy after sleep — you can move it by hand
- Green LED stays lit (Pi is on), the SDK exits cleanly

## What to watch for

- Any unusual sound from the servos (grinding, clicking)
- The head failing to return to centre
- The head sagging when you put it to sleep (this is normal — the motors are off)
- Any new smell (there shouldn't be one for a single 0.6 s move)

## If anything feels off

- Press `Ctrl+C` — the script will go to sleep in the `finally` block
- Or: `pkill -9 -f reachy-mini-daemon` to immediately cut the link

## What I will NOT do

- No antenna movement (those are the smallest servos, highest stall risk)
- No body yaw
- No held positions
- No infinite loops
- No `set_target` (non-blocking) calls
- No recorded moves from the dance library (those are 10+ s each)

---

**Reply with "go" (or any equivalent) and I will run the script.** The script is at `demos/probe_one_move.py` — let me create it now so you can review it before I run it.