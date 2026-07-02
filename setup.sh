#!/bin/bash
# =============================================================
# Meeting Transcriber — Setup Script
# =============================================================
# Run this once to install all dependencies.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${HOME}/.local/bin"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
BOOTSTRAP_PYTHON="${TRANSCRIPTION_BOOTSTRAP_PYTHON:-}"

if [ -z "$BOOTSTRAP_PYTHON" ]; then
    if command -v python3.12 &> /dev/null; then
        BOOTSTRAP_PYTHON="$(command -v python3.12)"
    elif [ -x "$HOME/miniforge3/bin/python3" ]; then
        BOOTSTRAP_PYTHON="$HOME/miniforge3/bin/python3"
    else
        BOOTSTRAP_PYTHON="$(command -v python3)"
    fi
fi

echo "========================================"
echo "🎙️  Meeting Transcriber — Setup"
echo "========================================"

# Check Python version
"$BOOTSTRAP_PYTHON" --version || { echo "❌ Python 3 is required."; exit 1; }

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo ""
    echo "⚠️  ffmpeg not found. Install it:"
    echo "   Fedora:  sudo dnf install ffmpeg"
    echo "   macOS:   brew install ffmpeg"
    echo "   Ubuntu:  sudo apt install ffmpeg"
    echo ""
    exit 1
fi

# Check for PipeWire/PulseAudio tools (needed for live capture)
if ! command -v pactl &> /dev/null; then
    echo ""
    echo "⚠️  pactl not found. Install PipeWire PulseAudio compat:"
    echo "   Fedora:  sudo dnf install pipewire-pulseaudio pulseaudio-utils"
    echo ""
fi

if ! command -v parec &> /dev/null; then
    echo ""
    echo "⚠️  parec not found. Install PulseAudio utils:"
    echo "   Fedora:  sudo dnf install pulseaudio-utils"
    echo ""
fi

echo ""
echo "🐍 Creating repo virtualenv..."
"$BOOTSTRAP_PYTHON" -m venv --clear "$VENV_DIR"

echo ""
echo "📦 Installing CPU-only PyTorch into $VENV_DIR..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall

echo ""
echo "📦 Installing Whisper runtime dependencies..."
"$VENV_PYTHON" -m pip install more-itertools numba numpy tiktoken tqdm

echo ""
echo "📦 Installing Whisper without optional GPU compiler dependency..."
"$VENV_PYTHON" -m pip install --no-deps openai-whisper

echo ""
echo "🔎 Checking Codex CLI for summarization..."
if command -v codex &> /dev/null; then
    echo "   codex: $(command -v codex)"
elif [ -x "$HOME/.local/bin/codex" ]; then
    echo "   codex: $HOME/.local/bin/codex"
else
    echo "⚠️  codex CLI not found. Install Codex CLI and run: codex login"
fi

echo ""
echo "🔗 Installing meeting launcher..."
mkdir -p "$BIN_DIR"
chmod +x "$SCRIPT_DIR/meeting"
ln -sfn "$SCRIPT_DIR/meeting" "$BIN_DIR/meeting"
echo "   $BIN_DIR/meeting -> $SCRIPT_DIR/meeting"

echo ""
echo "ℹ️  Diarization: Python Whisper uses pyannote.audio when --diarize is set."
echo "   This requires HF_TOKEN. Without --diarize, dual-channel mode still tags"
echo "   your mic as Me and system audio as Others."

echo ""
echo "========================================"
echo "✅ Setup complete!"
echo "========================================"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " LIVE CAPTURE (capture.py)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  List audio sources:"
echo "    .venv/bin/python capture.py"
echo ""
echo "  Record mic + system audio (Teams/Zoom/Meet):"
echo "    .venv/bin/python capture.py --record"
echo ""
echo "  Record + auto-transcribe when done:"
echo "    .venv/bin/python capture.py --record --auto"
echo ""
echo "  Full pipeline (CPU, summarize with local Codex CLI GPT-5.5 xhigh):"
echo "    .venv/bin/python capture.py --record --auto --summarize"
echo ""
echo "  Shortcut:"
echo "    meeting"
echo ""
echo "  Diarize other speakers too (requires HF_TOKEN with Python Whisper):"
echo "    meeting --diarize"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " CODEX SUMMARIZATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Codex CLI (default):"
echo "    --summarize"
echo "    --summarize --codex-model gpt-5.5 --reasoning-effort xhigh"
echo "    Setup: codex login"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " LOCAL LLAMA.CPP NOTE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  This repo uses Python Whisper for audio transcription."
echo "  Homebrew llama.cpp is installed locally for text/multimodal GGUF models,"
echo "  but it is not a drop-in replacement for whisper.cpp transcription."
echo "  Setup installs CPU-only PyTorch; no NVIDIA CUDA packages are required."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " SETUP NOTES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  For diarization, you need a HuggingFace token:"
echo "    1. Create account at https://huggingface.co"
echo "    2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "    3. Create token at https://huggingface.co/settings/tokens"
echo "    4. export HF_TOKEN=hf_your_token_here"
