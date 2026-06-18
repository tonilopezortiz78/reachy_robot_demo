# Audio pipeline — why the first attempt played through the laptop speakers

## The mistake

I started by playing the WAV with:

```bash
pactl set-sink-volume alsa_output.usb-Pollen_Robotics_Reachy_Mini_Audio_... 150%
ffmpeg -i hello.wav -f pulse -ac 1 -ar 16000 alsa_output.usb-Pollen_Robotics_Reachy_Mini_Audio_...  analog-stereo
```

The user said: *"the volume was low, but I hear it"*. Then: *"you raised the volume of my computer, not the robot"*. They were right.

## What was actually happening

PipeWire (the default audio system on Pop!_OS) was honouring the request to play to that PulseAudio sink — but `plughw:2,0` (the actual robot speaker) is the same physical device as `alsa_output.usb-Pollen_Robotics_...`. The volume I set was the **input gain to PipeWire's processing**, not the robot's amp. The audio path was:

```
Piper TTS → WAV file
    → ffmpeg (PulseAudio output, target sink = Reachy USB)
    → PipeWire mixes it with everything else
    → routes to the Reachy USB ALSA device
    → but the OS treats the same device as "your computer's USB speaker"
```

The user heard it because it did reach the robot, but the volume control was the wrong one. Also, the routing depended on the default sink and would have changed after any reboot or audio event.

## The fix — direct ALSA

```bash
aplay -D plughw:2,0 -f S16_LE -r 22050 -c 1 hello.wav
```

`plughw:2,0` is the **second sound card (`2`), device 0**, opened by the `plug` ALSA plugin (which handles sample-rate conversion). This:

- bypasses PipeWire/PulseAudio entirely
- goes straight to the robot's USB speaker endpoint
- is impossible to accidentally route to the laptop speakers
- plays only on the robot

In code:

```python
import subprocess
subprocess.Popen(["aplay", "-D", "plughw:2,0", "-q", wav_path])
```

## Setting the robot's actual volume

The robot's speaker has a hardware-side volume control exposed via the daemon's REST API:

```bash
curl -X POST http://localhost:8000/api/volume/set -H 'Content-Type: application/json' -d '{"volume": 100}'
```

(Requires the media server to be up — only available with `webrtcsink` installed.)

**For now**, just control the loudness in software: tell `aplay` / `ffmpeg` to write a louder signal. With the TTS already in `s16le` PCM, a software gain of `2.0×` is plenty.

## Audio format notes

- The robot's USB mic + speaker are 16-bit, 16 kHz, mono (per the UAC descriptor)
- Piper TTS output is 16-bit, 22050 Hz, mono
- ALSA's `plug` plugin resamples automatically — no need to pre-resample
