# Reachy Mini — Demo Reference

All demos run via `./run.sh demos/<file>.py` or through `./menu.sh`.

---

## Quick reference

| # | Demo | File | Services needed | Internet |
|---|---|---|---|---|
| 1 | Welcome | `demo_welcome.py` | — | No |
| 2 | Dance Show | `demo_dance.py` | — | No |
| 3 | NS Ambassador (Piper) | `demo_talk_ns.py` | Groq API | Yes |
| 4 | Face Recognition | `demo_face_recognition.py` | — | No |
| 5 | NS Ambassador (edge-tts) | `demo_edge.py` | Groq API, edge-tts | Yes |
| 6 | Fluid Dialog | `demo_dialog.py` | Groq API, edge-tts | Yes |
| 7 | LLM Tools | `demo_tools7.py` | Groq API, edge-tts | Yes |
| 8 | DeepSeek Flash | `demo_deepseek.py` | Groq API, edge-tts, opencode | Yes |

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

## Demo 3 — NS Ambassador, Piper voice (`demo_talk_ns.py`)

**What it does:** Full conversational demo. Reachy listens, understands, and replies about Network School, Virtuals Protocol, Quantus Protocol, Bitcoin, and AI. Replies in the user's language and can switch mid-conversation. Robot animates throughout.

**Character:** Reachy has no arms and no legs yet — just a head, antennas, and a rotating body — and acknowledges this with humor. Replies are kept to 1 sentence / 10 words max.

**Pipeline:**
```
Mic → Silero VAD → Groq Whisper STT → Groq LLaMA → Piper TTS → Robot speaker
```

**Tools / stack:**
- Silero VAD — voice activity detection, runs locally on CPU
- Groq API (`whisper-large-v3`) — speech-to-text (full model for best multilingual accuracy)
- Groq API (`meta-llama/llama-4-scout-17b-16e-instruct`) — language model
- Piper TTS (`en_US-amy-medium`) — local, fast (~50ms) offline synthesis
- ffmpeg — audio FX + processing
- aplay — robot speaker

**Requires internet** (Groq API).

**Knowledge base:** `docs/NS_KNOWLEDGE.md`

**API key:** `GROQ_API_KEY` in `.env` or environment.

---

## Demo 4 — Face Recognition (`demo_face_recognition.py`)

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

## Demo 5 — NS Ambassador, edge-tts voice (`demo_edge.py`)

**What it does:** Same personality and knowledge base as Demo 3, but uses a single Microsoft multilingual voice — `en-US-AvaMultilingualNeural` — for all languages instead of Piper. Pipelined synthesis hides latency — sentence N+1 is synthesized in the background while sentence N is playing.

**Voice:** `en-US-AvaMultilingualNeural` at RATE `+30%`, PITCH `+52Hz`, VOL `2.5`. AvaMultilingual is an adult voice at 0Hz; the `+52Hz` pitch lift is what makes it sound cute and young like a small robot. The same voice handles every language.

**Pipeline:**
```
Mic → Silero VAD → Groq Whisper STT → Groq LLaMA → edge-tts → Robot speaker
```

**Tools / stack:**
- Silero VAD — voice activity detection (local)
- Groq API (`whisper-large-v3`) — speech-to-text (full model for best multilingual accuracy)
- Groq API (`meta-llama/llama-4-scout-17b-16e-instruct`) — language model
- edge-tts (`en-US-AvaMultilingualNeural`, pitch `+52Hz`) — synthesis for any language, Microsoft cloud
- ffmpeg — resample + volume
- aplay — robot speaker

**Requires internet** (Groq API + edge-tts).

**vs Demo 3:** Better sounding voice that works in any language. Requires network for synthesis (Demo 3's Piper voice is offline). Slightly higher latency to first word (~400ms more) but no gaps between sentences.

---

## Demo 6 — Fluid Dialog (`demo_dialog.py`)

**What it does:** Fast conversational demo with barge-in support (interrupt Reachy mid-sentence). ~700ms turn-taking with high-threshold VAD during TTS to avoid echo. Works in any language and switches mid-conversation.

**Pipeline:**
```
Mic → Silero VAD → Groq Whisper STT → Groq LLaMA → edge-tts → Robot speaker
```

**Tools / stack:**
- Silero VAD — voice activity detection with barge-in (local)
- Groq API (`whisper-large-v3`) — speech-to-text (full model for best multilingual accuracy)
- Groq API (`meta-llama/llama-4-scout-17b-16e-instruct`) — language model
- edge-tts (`en-US-AvaMultilingualNeural`, pitch `+52Hz`) — synthesis for any language, Microsoft cloud
- ffmpeg — resample + volume
- aplay — robot speaker

**Requires internet** (Groq API + edge-tts).

---

## Demo 7 — LLM Tools (`demo_tools7.py`)

**What it does:** The most advanced Groq-based talking demo. Same NS ambassador personality as Demos 3/5, but adds barge-in and a parallel AI gesture picker — while the LLM streams its reply, a second call selects a fitting gesture so motion and speech stay in sync. Works in any language. Includes session logging, long-term memory, hallucination rejection, and spoken cues in the user's language.

**Pipeline:**
```
Mic → Silero VAD → Groq Whisper STT → Groq LLaMA ┬→ edge-tts → Robot speaker
                                                  └→ Groq LLaMA (gesture pick) → motion
```

**Tools / stack:**
- Silero VAD — voice activity detection with barge-in (local)
- Groq API (`whisper-large-v3`) — speech-to-text (full model for best multilingual accuracy)
- Groq API (`meta-llama/llama-4-scout-17b-16e-instruct`) — language model + parallel gesture picker
- edge-tts (`en-US-AvaMultilingualNeural`, pitch `+52Hz`) — synthesis for any language, Microsoft cloud
- ffmpeg — resample + volume
- aplay — robot speaker

**Requires internet** (Groq API + edge-tts).

---

## Demo 8 — DeepSeek Flash (`demo_deepseek.py`)

**What it does:** Same as Demo 7 (barge-in, parallel gesture picker, session logging, memory, multilingual) but uses `opencode run` as the LLM harness instead of calling Groq's LLM API directly. opencode's default model (DeepSeek V4 Flash) powers all text generation. ~8s LLM latency (opencode overhead) — thinking ticks and spoken cues cover the gap. STT still via Groq Whisper.

**Pipeline:**
```
Mic → Silero VAD → Groq Whisper STT → opencode run (DeepSeek V4 Flash) ┬→ edge-tts → Robot speaker
                                                                       └→ opencode (gesture pick) → motion
```

**Tools / stack:**
- Silero VAD — voice activity detection with barge-in (local)
- Groq API (`whisper-large-v3`) — speech-to-text (full model for best multilingual accuracy)
- opencode CLI (`opencode run`, default model DeepSeek V4 Flash) — language model + gesture picker
- edge-tts (`en-US-AvaMultilingualNeural`, pitch `+52Hz`) — synthesis for any language, Microsoft cloud
- ffmpeg — resample + volume
- aplay — robot speaker

**Requires internet** (Groq API + edge-tts). opencode's default model must be DeepSeek V4 Flash.

---

## Environment setup

**API key** (required for demos 3, 5, 6, 7, 8):
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
