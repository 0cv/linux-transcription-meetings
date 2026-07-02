# Linux Meeting Transcriber

Capture and transcribe meeting audio on Linux. Records your microphone and system audio (Teams/Zoom/Meet) as separate tracks, transcribes with Whisper, identifies speakers, and generates Obsidian-ready meeting notes.

## How It Works

1. **Capture** — Records mic and system audio simultaneously via PipeWire/PulseAudio as separate WAV files
2. **Transcribe** — Runs Python Whisper on each track
3. **Tag speakers** — Your mic is always labeled "Me"; other speakers are diarized as "Speaker 1", "Speaker 2", etc.
4. **Summarize** — Optionally generates structured meeting notes through the local Codex CLI using GPT-5.5 xhigh reasoning
5. **Output** — Obsidian-compatible markdown with YAML frontmatter, plus plain-text transcript

## Requirements

- **Linux** with PipeWire or PulseAudio (default on Fedora)
- **Python 3.10+**
- **ffmpeg**, **pactl**, **parec** (from `pulseaudio-utils`)

## Setup

```bash
./setup.sh
```

This creates a repo-local `.venv`, installs local Python Whisper (`openai-whisper`) with CPU-only PyTorch, and adds a `meeting` launcher in `~/.local/bin`. It does not require NVIDIA CUDA.

Log in to Codex CLI before using summaries:

```bash
codex login
```

`whisper.cpp` was removed from this machine. The local Homebrew `llama.cpp` install is useful for GGUF text/multimodal models, but it is not a drop-in audio transcription backend for this repo.

## Usage

### List audio sources

```bash
.venv/bin/python capture.py
```

### Record and transcribe

```bash
# Shortcut: record, transcribe, and summarize with local Codex CLI GPT-5.5 xhigh
meeting

# Record mic + system audio, stop with Ctrl+C
.venv/bin/python capture.py --record

# Full pipeline: record → transcribe → diarize → summarize
.venv/bin/python capture.py --record --auto --diarize --summarize

# Explicit Codex CLI summary settings
.venv/bin/python capture.py --record --auto --summarize --codex-model gpt-5.5 --reasoning-effort xhigh
```

### Transcribe existing files

```bash
# Dual-channel (recommended)
.venv/bin/python transcribe.py --mic mic.wav --system system.wav --diarize --summarize

# Single file
.venv/bin/python transcribe.py meeting.wav --diarize --summarize
```

### Specify audio sources manually

```bash
.venv/bin/python capture.py --record --mic 5 --monitor 3
```

## Summarization

Summaries use only the local Codex CLI through `codex exec`.

Defaults:
- Model: `gpt-5.5`
- Reasoning effort: `xhigh`

## Vocabulary Dictionary

Edit `dictionary.txt` to add names, products, and domain terms. These are:
- Fed to Whisper as a prompt to bias the decoder toward correct spellings
- Included in the Codex summary prompt for post-correction

## Diarization

Python Whisper diarization uses [pyannote.audio](https://github.com/pyannote/pyannote-audio). It requires:
  1. Accept terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  2. Set `HF_TOKEN` environment variable

## Automatic Meeting Name Detection

When you start recording without `--name`, the tool scans your window titles for an active Teams, Zoom, or Google Meet meeting and names the recording + notes accordingly (e.g., `2026-04-02_weekly-standup.md`).

Requires `kdotool` (KDE Plasma on Wayland):

```bash
sudo dnf install kdotool
```

Use `--no-detect` to skip the lookup, or `--name "my-meeting"` to set it manually.

## Output

WAV recordings are saved to `/tmp/meetings` by default (override with `--recordings-dir`).

Obsidian notes are saved to `~/Documents/notes2/meetings` by default (override with `--output`):

- `YYYY-MM-DD_meeting-name.md` — Obsidian note with summary + full transcript
- `YYYY-MM-DD_meeting-name_transcript.txt` — Plain-text transcript
- `YYYY-MM-DD_meeting-name_transcript.json` — Raw segments (with `--json`)
