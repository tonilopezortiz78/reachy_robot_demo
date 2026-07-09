# Two-Reachy Demo — Future Plan

Status: **blocked on hardware** (no second robot yet). Saved 2026-07-09.
Revive when a second Reachy Mini Lite is available.

## Concept: one talker + one mute reactor (comedy duo for kids)

Robot A is the full talking robot (mic, speaker, LLM, face ID, dashboard).
Robot B is a **mute motor puppet** that reacts to A — nods, shakes head, looks
confused, "cries" when ignored, celebrates when a kid picks it, dances in the
Macarena. One talker + one reactor is the low-risk path because it sidesteps
the unsolved dual-audio problem (see *Risks*).

## Why this split (not two talkers)

- The SDK supports two motors/daemons/cameras out of the box.
- The audio layer does NOT: both robots' mics and speakers collide on the ALSA
  card name `Reachy_Mini_Audio` and the PipeWire source substring
  `Reachy_Mini_Audio`. `_detect_robot_mic` returns the first match; there is no
  per-unit selector. Solving this is unproven work — wrong thing to risk before
  a live show.
- One talker + one puppet = zero audio work on B. B becomes a motor-only actor.

## SDK feasibility (verified 2026-07-09)

| Subsystem | Two-robot support | Notes |
|---|---|---|
| Motors / daemon | Yes | `reachy-mini-daemon --serialport /dev/ttyACM0 --fastapi-port 8000` and `--serialport /dev/ttyACM1 --fastapi-port 8001`. Daemon's `find_serial_port` raises on >1 port unless `--serialport` is explicit (daemon/utils.py). |
| SDK constructor | Yes | `ReachyMini(connection_mode="localhost_only", port=8001)` targets the second daemon. No `device` param — port-to-USB mapping lives in the daemon. Two instances in one process are fine (each builds its own WSClient). |
| Camera | Yes | `CameraHub("/dev/video4")` — already parameterized (camera.py:25,30). |
| Speaker | Hardcoded | `SPEAKER = "plughw:CARD=Audio,DEV=0"` (audio.py:30). Second unit gets ALSA id `Audio_1` but no code targets it. |
| Mic | First-wins | `_MIC_PREFERENCE = ("Reachy_Mini_Audio", ...)` returns first match (audio.py:54,104-107). Both sources contain the substring. Disambiguation by per-unit serial in the PipeWire name is *possible* but not implemented. |

USB with two robots: `/dev/ttyACM0`+`/dev/ttyACM1` (motors),
`/dev/video2-3`+`/dev/video4-5` (cams), two ALSA cards both named "Reachy Mini
Audio" (ids `Audio` + `Audio_1`).

## Run-of-show (10-min slot inside the 15-min demo)

1. Reachy A greets, learns a kid's name, chats (the proven solo show).
2. **Reachy B sulks** — turns away, antennas droop, looks sad.
3. A & B "argue" — A speaks, B shakes head; A `[celebrate]`, B `[confused]`.
4. A kid asks B a question → A "translates" for B → B nods enthusiastically.
5. **Dance duet** — both do the Macarena, B deliberately out of sync (goofy).
6. B "finally speaks" — A voices B's line, B moves body in sync. Kids lose it.

## What to build when hardware arrives

- `reachy_demo/reactor.py` — `ReactorBot` class: takes `mini_b` + shared
  `LiveState` + `anim_b`; subscribes to state changes. Rules:
  - A `SPEAKING` → B looks at the kid.
  - A played `[celebrate]` → B plays `[confused]` or `[shy]`.
  - A dancing → B dances, beat-offset (~1 beat behind = goofy).
  - Idle > 10s → B snoozes (slow antenna droop + head nod).
- `demo_hackathon.py` (or extend it) — construct second daemon on :8001 +
  `ReachyMini(port=8001)` + `ReactorBot`. Reuses `ConverseEngine` from
  `demo_converse.py` for A.
- Dashboard control panel gets a "make B react" puppet panel (trigger gestures
  on B live).

## Risks to solve before a two-talker show (post-hackathon, optional)

- Mic disambiguation by PipeWire source serial (rewrite `_detect_robot_mic` to
  take a robot selector).
- Speaker routing to `CARD=Audio_1` (parameterize `SPEAKER`, thread through
  `play_wav_blocking`, `_beep`, `tts_edge`).
- Cross-talk: two mics in one room hearing the same kid — need per-robot VAD
  gating or directional selection.
- Two `ReachyMini` cleanup: both must `goto_sleep()` in `finally` (overheat
  risk doubles).

## Pre-flight checklist when reviving

- [ ] Second Reachy Mini Lite hardware in hand + second USB-C cable.
- [ ] `lsusb` shows two QinHeng `1a86:55d3` bridges; `arecord -l` shows two
      "Reachy Mini Audio" cards; `/dev/video2` and `/dev/video4` both present.
      (Recipe in `docs/HARDWARE_DIAGNOSIS.md`.)
- [ ] Launch two daemons manually with explicit `--serialport` + `--fastapi-port`.
- [ ] Smoke test: two `ReachyMini(port=...)` objects, both `wake_up()` +
      `goto_sleep()`, no motor errors.
- [ ] Then build `reactor.py` + extend `demo_hackathon.py`.
