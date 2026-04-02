# Linux Meeting Transcriber

Capture and transcribe meeting audio on Linux. Records your microphone and system audio (Teams/Zoom/Meet) as separate tracks, transcribes with Whisper, identifies speakers, and generates Obsidian-ready meeting notes.

## How It Works

1. **Capture** — Records mic and system audio simultaneously via PipeWire/PulseAudio as separate WAV files
2. **Transcribe** — Runs Whisper (CPU or GPU via whisper.cpp + Vulkan) on each track
3. **Tag speakers** — Your mic is always labeled "Me"; other speakers are diarized as "Speaker 1", "Speaker 2", etc.
4. **Summarize** — Optionally generates structured meeting notes via Claude, Ollama, or OpenAI
5. **Output** — Obsidian-compatible markdown with YAML frontmatter, plus plain-text transcript

## Requirements

- **Linux** with PipeWire or PulseAudio (default on Fedora)
- **Python 3.10+**
- **ffmpeg**, **pactl**, **parec** (from `pulseaudio-utils`)

## Setup

```bash
./setup.sh
```

This installs OpenAI Whisper, CPU-only PyTorch, and summarization libraries (anthropic, ollama, openai).

### Optional: GPU acceleration with whisper.cpp + Vulkan

For AMD GPUs (no ROCm needed):

```bash
sudo dnf install vulkan-loader vulkan-headers vulkan-tools vulkan-validation-layers

cd ~ && git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build -DGGML_VULKAN=1
cmake --build build -j
bash models/download-ggml-model.sh large-v3-turbo
```

Verify your GPU is detected: `vulkaninfo --summary`

## Usage

### List audio sources

```bash
python capture.py
```

### Record and transcribe

```bash
# Record mic + system audio, stop with Ctrl+C
python capture.py --record

# Full pipeline: record → transcribe → diarize → summarize
python capture.py --record --auto --diarize --summarize

# GPU-accelerated
python capture.py --record --auto --engine whisper-cpp --diarize --summarize

# Summarize locally with Ollama instead of Claude
python capture.py --record --auto --diarize --summarize --llm ollama
```

### Transcribe existing files

```bash
# Dual-channel (recommended)
python transcribe.py --mic mic.wav --system system.wav --diarize --summarize

# Single file
python transcribe.py meeting.wav --diarize --summarize
```

### Specify audio sources manually

```bash
python capture.py --record --mic 5 --monitor 3
```

## Summarization Backends

| Backend | Flag | Auth | Notes |
|---------|------|------|-------|
| **Claude** (default) | `--llm claude` | `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY` | Uses `claude` CLI pipe mode or SDK |
| **Ollama** | `--llm ollama` | None (local) | Default model: `qwen3:8b`. Install from [ollama.com](https://ollama.com) |
| **OpenAI** | `--llm openai` | `OPENAI_API_KEY` | Default model: `gpt-4o-mini` |

## Vocabulary Dictionary

Edit `dictionary.txt` to add names, products, and domain terms. These are:
- Fed to Whisper as a prompt to bias the decoder toward correct spellings
- Included in the LLM summary prompt for post-correction

## Diarization

Two approaches depending on engine:

- **whisper-cpp** — Uses built-in tinydiarize (`-tdrz`). No extra setup needed.
- **whisper (Python)** — Uses [pyannote.audio](https://github.com/pyannote/pyannote-audio). Requires:
  1. Accept terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  2. Set `HF_TOKEN` environment variable

## Output

Files are saved to `~/Documents/notes2/meetings` by default (override with `--output`):

- `YYYY-MM-DD_meeting-name.md` — Obsidian note with summary + full transcript
- `YYYY-MM-DD_meeting-name_transcript.txt` — Plain-text transcript
- `YYYY-MM-DD_meeting-name_transcript.json` — Raw segments (with `--json`)
