# Reachy Mini — Demo Reference

All demos run via `./run.sh demos/<file>.py` or through `./menu.sh`.

---

## Quick reference

| # | Demo | File | Services needed | Internet |
|---|---|---|---|---|
| 1 | Welcome | `demo_welcome.py` | — | No |
| 2 | Dance Show | `demo_dance.py` | — | No |
| 3 | Face Tracking | `demo_face.py` | — | No |
| 4 | Lost Brother | `demo_lost_friend.py` | — | No |
| 5 | NS Ambassador (Piper) | `demo_talk_ns.py` | Groq API | Yes |
| 6 | Free Chat | `demo_chat.py` | Groq API | Yes |
| 7 | Face Recognition | `demo_face_recognition.py` | — | No |
| 8 | NS Ambassador (edge-tts) | `demo_edge.py` | Groq API, Microsoft edge-tts | Yes |

---

## Demo 1 — Welcome (`demo_welcome.py`)

**What it does:** Reachy boots up, speaks a welcome line about Network School, and holds an attentive listening pose.

**Runtime:** ~25 seconds

**Tools / stack:**
- Piper TTS (`voices/en_US-amy-medium.onnx`) — local, offline
- ffmpeg — audio FX + playback
- aplay — robot speaker output
- Reachy Mini SDK — motion

**No internet required.**

---

## Demo 2 — Full Dance Show (`demo_dance.py`)

**What it does:** Boot sequence → greeting speech → music starts → ~60s curated dance using preloaded `RecordedMoves` presets → victory bow.

**Runtime:** ~90 seconds

**Tools / stack:**
- Piper TTS — greeting speech
- ffmpeg / aplay — music + audio
- `RecordedMoves` from HuggingFace (`pollen-robotics/reachy-mini-dances-library`, `pollen-robotics/reachy-mini-emotions-library`) — preloaded dance presets
- Reachy Mini SDK — motion

**No internet required** (HuggingFace models are cached locally on first run).

**Swap the music:** edit `MUSIC = ...` at the top of the file. Drop any MP3/WAV in `music/`.

---

## Demo 3 — Face Tracking (`demo_face.py`)

**What it does:** Camera opens immediately. Once robot is ready, the head, body, and antennas track any detected face in real time. Live preview window with overlay.

**Tools / stack:**
- OpenCV (`cv2`) — camera capture + Haar cascade face detection
- Reachy Mini SDK — `set_target()` at 20 Hz

**No internet required.**

**Camera:** `/dev/video2` (UVC, 640×360).

---

## Demo 4 — Lost Brother (`demo_lost_friend.py`)

**What it does:** Scripted emotional monologue — Reachy tells the story of its lost robot brother Pixel and pitches the idea of an NS Robotics Club.

**Runtime:** ~3 minutes

**Tools / stack:**
- Piper TTS + ffmpeg — speech with robot FX
- `RecordedMoves` — emotion presets (`sad1`, `worried`, `loving1`, etc.)
- Reachy Mini SDK — motion + presets

**No internet required.**

---

## Demo 5 — NS Ambassador, Piper voice (`demo_talk_ns.py`)

**What it does:** Full conversational demo. Reachy listens, understands, and replies about Network School, Virtuals Protocol, Quantus Protocol, Bitcoin, and AI. Supports English and Mandarin Chinese. Robot animates throughout.

**Pipeline:**
```
Mic → Silero VAD → Groq Whisper STT → Groq LLaMA → Piper TTS → Robot speaker
```

**Tools / stack:**
- Silero VAD — voice activity detection, runs locally on CPU
- Groq API (`whisper-large-v3-turbo`) — speech-to-text
- Groq API (`meta-llama/llama-4-scout-17b-16e-instruct`) — language model
- Piper TTS (`en_US-amy-medium`) — English synthesis, local, fast (~50ms)
- edge-tts (`zh-CN-YunyangNeural`) — Mandarin synthesis, Microsoft cloud
- ffmpeg — audio FX + processing
- aplay — robot speaker

**Requires internet** (Groq API + edge-tts for Chinese).

**Knowledge base:** `docs/NS_KNOWLEDGE.md`

**API key:** `GROQ_API_KEY` in `.env` or environment.

---

## Demo 6 — Free Chat (`demo_chat.py`)

**What it does:** Open conversational loop. Same voice pipeline as Demo 5 but with a simpler, general-purpose persona. English only (Whisper forced to `language="en"`).

**Pipeline:**
```
Mic → Silero VAD → Groq Whisper STT → Groq LLaMA → Piper TTS → Robot speaker
```

**Tools / stack:**
- Same as Demo 5, minus Chinese TTS and NS knowledge base
- Model: `llama-3.3-70b-versatile`

**Requires internet** (Groq API).

---

## Demo 7 — Face Recognition (`demo_face_recognition.py`)

**What it does:** Loads a roster of known faces from `faces/<name>/*.jpg`. When a known person enters frame, Reachy greets them by name. Unknown visitors get a generic welcome. Head tracks the face in real time. 90-second cooldown between greetings.

**Tools / stack:**
- `face_recognition` (dlib ResNet) — face embedding + matching, runs on CPU
- OpenCV — camera capture + display overlay
- Piper TTS — greeting speech
- ffmpeg / aplay — audio
- Reachy Mini SDK — face tracking motion

**No internet required.**

**Roster setup:**
```bash
mkdir -p faces/tony
cp your_photo.jpg faces/tony/
# Multiple photos per person = more reliable
```

**Tolerance:** 0.52 (lower = stricter). Edit `TOLERANCE` at the top of the file.

---

## Demo 8 — NS Ambassador, edge-tts voice (`demo_edge.py`)

**What it does:** Identical personality and knowledge base to Demo 5, but uses Microsoft `en-US-AriaNeural` for English (natural, warm, expressive) instead of Piper. Chinese still uses `zh-CN-YunyangNeural`. Pipelined synthesis hides latency — sentence N+1 is synthesized in the background while sentence N is playing.

**Pipeline:**
```
Mic → Silero VAD → Groq Whisper STT → Groq LLaMA → edge-tts → Robot speaker
```

**Tools / stack:**
- Silero VAD — voice activity detection (local)
- Groq API (`whisper-large-v3-turbo`) — speech-to-text
- Groq API (`meta-llama/llama-4-scout-17b-16e-instruct`) — language model
- edge-tts (`en-US-AriaNeural`) — English synthesis, Microsoft cloud
- edge-tts (`zh-CN-YunyangNeural`) — Mandarin synthesis, Microsoft cloud
- ffmpeg — resample + volume
- aplay — robot speaker

**Requires internet** (Groq API + edge-tts).

**vs Demo 5:** Better sounding voice. Requires network for English too (Demo 5 English is offline). Slightly higher latency to first word (~400ms more) but no gaps between sentences.

---

## Environment setup

**API key** (required for demos 5, 6, 8):
```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

**Hardware switch:** must be in **Robot / Developer** position (not "Computer").

**Orphaned daemon** (if a demo was killed with kill -9):
```bash
pkill -9 -f reachy-mini-daemon
```

**Voices directory** — Piper model (61 MB, not in repo):
```bash
cd voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
```
