# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A meeting transcription pipeline for Linux (Fedora/PipeWire). Two scripts:

- **`capture.py`** — Records audio from system monitor (meeting audio) and/or microphone via `parec` → `ffmpeg`. Dual-source mode records both as separate WAV files, then merges into a stereo file (L=mic, R=system). WAVs go to `/tmp/meetings`; notes go to the Obsidian vault.
- **`transcribe.py`** — Transcribes audio with Whisper, optionally diarizes speakers, summarizes with an LLM, and outputs Obsidian-formatted meeting notes.
- **`meeting_detect.py`** — Auto-detects the current meeting name by scanning window titles (Teams/Zoom/Meet) via `kdotool` on KDE Wayland.

## Setup

```bash
./setup.sh   # installs all Python deps + checks for ffmpeg, pactl, parec
```

## Common Commands

```bash
# List audio sources
python capture.py

# Record (mic + system), then auto-transcribe with diarization and summary
python capture.py --record --auto --diarize --summarize

# GPU-accelerated via whisper.cpp + Vulkan
python capture.py --record --auto --engine whisper-cpp --diarize --summarize

# Transcribe existing files (dual-channel)
python transcribe.py --mic mic.wav --system system.wav --diarize --summarize

# Transcribe single file
python transcribe.py meeting.wav

# Local summarization (no cloud)
python capture.py --record --auto --diarize --summarize --llm ollama
```

## Architecture

### Audio Capture (`capture.py`)

Uses `pactl` for source discovery and `parec` piped into `ffmpeg` for recording. Three recorder classes:
- `ParecRecorder` — records a single PulseAudio/PipeWire source to WAV
- `DualRecorder` — runs two `ParecRecorder` instances in parallel (mic + system monitor), merges output into stereo WAV
- `SingleRecorder` — wraps `ParecRecorder` for system-only or mic-only mode

All audio is captured at 16kHz mono (per-track). On stop, `DualRecorder` merges via `ffmpeg amerge` into stereo. WAV files are stored in `--recordings-dir` (default `/tmp/meetings`), separate from the Obsidian notes directory (`--output`).

If no `--name` is given, `capture.py` scans window titles via `meeting_detect.get_current_meeting()` to auto-name the recording after the active Teams/Zoom/Meet meeting. Falls back to `meeting_{timestamp}` if no meeting window is found.

### Transcription (`transcribe.py`)

Two engine backends:
- **`whisper`** — OpenAI's Python whisper package, CPU-only. Diarization via `pyannote.audio` (requires HF_TOKEN).
- **`whisper-cpp`** — whisper.cpp CLI with Vulkan GPU acceleration. Diarization via `-tdrz` flag (tinydiarize, no external deps). Expects `~/whisper.cpp` or `WHISPER_CPP_DIR` env var.

Dual-channel transcription (`transcribe_dual_channel`) transcribes mic and system tracks independently, tags mic as "Me", then merges timelines chronologically. Echo duplicates (your voice leaking into system audio) are filtered by time-overlap + text-similarity heuristics in `_remove_echo_duplicates`.

Three summarization backends:
- **Claude** (default) — via `claude` CLI pipe mode (`-p`) using OAuth token, or `anthropic` SDK with API key
- **Ollama** — local models (default: `qwen3:8b`)
- **OpenAI** — via `openai` SDK

Output: Obsidian-compatible markdown notes with YAML frontmatter, plus plain-text transcript. Optionally JSON.

### Dictionary (`dictionary.txt`)

Domain-specific vocabulary (names, products, terms). Loaded automatically from the script directory. Fed to whisper as `--prompt` to bias the decoder and included in the LLM summary prompt for post-correction of misspellings.

## Key Details

- Default recordings directory: `/tmp/meetings` (WAV files)
- Default notes directory: `~/Documents/notes2/meetings` (Obsidian markdown)
- whisper.cpp model files are auto-downloaded if missing via the bundled `download-ggml-model.sh` script
- Claude auth precedence: `CLAUDE_CODE_OAUTH_TOKEN` → `ANTHROPIC_API_KEY` → `claude` CLI on PATH
- Meeting name detection requires `kdotool` (`sudo dnf install kdotool`) for KDE Plasma on Wayland
- pyannote diarization requires accepting terms at `huggingface.co/pyannote/speaker-diarization-3.1` and setting `HF_TOKEN`
