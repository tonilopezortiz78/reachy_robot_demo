# Camera pipeline — direct UVC capture without GStreamer

## The SDK can't see the camera because of one missing plugin

When the SDK daemon starts with media enabled, it spins up a GStreamer pipeline that ends in `webrtcsink` (a WebRTC sink implemented as a GStreamer Rust plugin). On a clean laptop that plugin isn't installed, so the media server fails:

```
No camera found.
Failed to create webrtcsink element. Is the GStreamer webrtc rust plugin installed?
```

The camera itself is fine. It's a standard UVC device on `/dev/video2`. We just bypass the SDK's pipeline.

## The direct path

```bash
ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 \
       -i /dev/video2 -frames:v 1 out.jpg
```

This opens the UVC device, grabs one frame, encodes it as JPEG, writes to disk. No SDK, no GStreamer, no plugin.

## Properties of the camera

```
$ v4l2-ctl -d /dev/video2 --info
Driver Info:
    Driver name      : uvcvideo
    Card type        : Reachy Mini Camera: Reachy Mini
    Bus info         : usb-0000:00:14.0-4.4
    Capabilities     : Video Capture, Metadata Capture, Extended Pix Format
```

ffmpeg will pick the camera's native size (1920×1080) when `-video_size` isn't strictly honoured. The metadata stream is at `/dev/video3`.

## A live preview without writing files

```bash
ffplay -f v4l2 -framerate 30 -video_size 1280x720 /dev/video2
```

This opens a window. Useful for aiming the robot.

## Using OpenCV in Python

```python
import cv2
cap = cv2.VideoCapture('/dev/video2', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ok, frame = cap.read()
```

`cv2` is in the standard `reachy_mini[opencv]` extra. Install it with `pip install opencv-python` if you want to use it directly.

## When to bother installing the GStreamer plugin

Only if you want:

- the SDK's own `mini.media.get_frame()` to work (used by `take_picture.py`, `look_at_image.py`)
- the SDK's own `mini.media.play_sound()` to work (used by `sound_play.py`, `sound_tts.py`)
- the browser apps (JS) to work over WebRTC

For everything in this repo, the direct paths above are simpler and more reliable.
