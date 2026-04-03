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

echo "========================================"
echo "🎙️  Meeting Transcriber — Setup"
echo "========================================"

# Check Python version
python3 --version || { echo "❌ Python 3 is required."; exit 1; }

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
echo "📦 Installing core dependencies..."
pip install --upgrade pip
pip install openai-whisper

echo ""
echo "📦 Installing CPU-only PyTorch (avoids CUDA errors on AMD systems)..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall

echo ""
echo "📦 Installing live capture dependencies..."
pip install sounddevice numpy scipy

echo ""
echo "📦 Installing calendar integration dependencies..."
pip install msal requests

echo ""
echo "📦 Installing summarization support..."
pip install anthropic              # Claude (default — uses your Claude Code OAuth token)
pip install ollama                 # Qwen 3 / Llama via Ollama (fully local alternative)
pip install openai                 # Needed for OpenAI fallback backend

echo ""
echo "ℹ️  Diarization: whisper.cpp --engine whisper-cpp --diarize uses built-in"
echo "   speaker turn detection (-tdrz). No extra dependencies needed."
echo "   (pyannote.audio is only needed if using --engine whisper --diarize)"

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
echo "    python capture.py"
echo ""
echo "  Record mic + system audio (Teams/Zoom/Meet):"
echo "    python capture.py --record"
echo ""
echo "  Record + auto-transcribe when done:"
echo "    python capture.py --record --auto"
echo ""
echo "  Full pipeline (CPU, summarize with Claude):"
echo "    python capture.py --record --auto --diarize --summarize"
echo ""
echo "  Full pipeline (GPU via whisper.cpp, summarize with Claude):"
echo "    python capture.py --record --auto --engine whisper-cpp --diarize --summarize"
echo ""
echo "  Full pipeline (summarize with local Qwen 3 via Ollama):"
echo "    python capture.py --record --auto --diarize --summarize --llm ollama"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " SUMMARIZATION BACKENDS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Claude (default) — uses your subscription via claude CLI:"
echo "    --summarize                          # Sonnet (default)"
echo "    --summarize --llm-model claude-opus-4-20250514  # Opus"
echo "    Setup:  claude setup-token"
echo "            export CLAUDE_CODE_OAUTH_TOKEN=<token>"
echo "    Or use an API key:  export ANTHROPIC_API_KEY=sk-ant-..."
echo ""
echo "  Ollama (fully local) — Qwen 3 or any local model:"
echo "    --summarize --llm ollama             # Qwen 3 8B (default)"
echo "    --summarize --llm ollama --llm-model qwen3:32b  # larger"
echo "    Requires: https://ollama.com → ollama pull qwen3:8b"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " WHISPER.CPP (GPU-ACCELERATED VIA VULKAN)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  For GPU-accelerated transcription on AMD (no ROCm needed):"
echo ""
echo "    # Install Vulkan SDK"
echo "    sudo dnf install vulkan-loader vulkan-headers vulkan-tools vulkan-validation-layers"
echo ""
echo "    # Build whisper.cpp with Vulkan"
echo "    cd ~ && git clone https://github.com/ggml-org/whisper.cpp"
echo "    cd whisper.cpp"
echo "    cmake -B build -DGGML_VULKAN=1"
echo "    cmake --build build -j"
echo ""
echo "    # Download the Turbo model"
echo "    bash models/download-ggml-model.sh large-v3-turbo"
echo ""
echo "    # Use it:"
echo "    python transcribe.py --engine whisper-cpp meeting.wav"
echo "    python capture.py --record --auto --engine whisper-cpp --diarize --summarize"
echo ""
echo "  Verify Vulkan sees your GPU:  vulkaninfo --summary"
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
