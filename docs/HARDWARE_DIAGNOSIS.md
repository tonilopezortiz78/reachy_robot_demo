# Hardware diagnosis — how I figured out what's where

This is a log of how I confirmed each of the robot's subsystems.

## Step 1: `lsusb` before and after plugging the robot in

Before: only your laptop's built-in devices (webcam, fingerprint reader, Bluetooth, etc.).

After: new devices appear on the USB bus:

```
Bus 001 Device 034: ID 1a86:55d3 QinHeng Electronics USB Single Serial   # motor serial bridge (CH340 clone)
Bus 001 Device 035: ID 38fb:1002 SunplusIT Inc Reachy Mini Camera        # camera
Bus 001 Device 036: ID 38fb:1001 Pollen Robotics Reachy Mini Audio       # speaker + mic
```

Three independent devices, all from Pollen's USB vendor ID `38fb`. None of them is a network interface, which tells you the Lite variant (no RNDIS/CDC-ECM).

## Step 2: serial port appears

`ls /dev/ttyACM*` shows `/dev/ttyACM0` once the robot is plugged in. The motors speak Feetech protocol over this.

## Step 3: audio device appears

```
$ arecord -l
card 1: Camera [Reachy Mini Camera], device 0: USB Audio [USB Audio]
card 2: Audio   [Reachy Mini Audio],   device 0: USB Audio [USB Audio]

$ aplay -l
card 0: PCH [HDA Intel PCH], ...      # laptop built-in
card 2: Audio [Reachy Mini Audio], device 0: USB Audio [USB Audio]   # robot speaker
```

So the robot's USB audio is **card 2**. ALSA direct device: `plughw:2,0`.

## Step 4: camera device appears

```
$ v4l2-ctl --list-devices
Reachy Mini Camera: Reachy Mini (usb-0000:00:14.0-4.4):
        /dev/video2
        /dev/video3          # metadata stream
```

`/dev/video2` is the main UVC stream. `v4l2-ctl -d /dev/video2 --info` confirms `Driver: uvcvideo`, `Bus info: usb-0000:00:14.0-4.4` — direct USB, not a network device.

## Step 5: confirm the camera works without the SDK

```bash
ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video2 -frames:v 1 reachy_cam_test.jpg
```

Got a valid 1920×1080 JPEG (ffmpeg picked the camera's native size). View confirms the camera is alive at the hardware level — the SDK just doesn't use it directly because of a missing GStreamer plugin.

## Step 6: the back switch is not a power switch

When you flip the switch, `lsusb` does **not** re-enumerate. The green LED stays on in both positions. The difference:

| Position | What the Pi does | SDK can drive motors? |
|---|---|---|
| Robot / Developer | Runs the control stack, exposes motors/audio/camera | Yes |
| Computer | Boots to a desktop, outputs to internal display | No (motors unresponsive) |

The user must flip to Robot/Developer for the SDK to work.

## Step 7: the green LED is the Pi's power/activity

Solid or slow blink = healthy. Off = no power (cable issue, switch in middle, or Pi is in a halt state).

## Step 8: mDNS check distinguishes Lite vs Wireless

```
$ ping -c1 reachy-mini.local
ping: reachy-mini.local: Name or service not known
```

Wireless units advertise themselves over mDNS. The Lite does not. The absence of `reachy-mini.local` is the easiest way to confirm you have the Lite.
