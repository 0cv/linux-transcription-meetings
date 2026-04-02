#!/usr/bin/env python3
"""
Live Audio Capture for Meeting Transcriber
===========================================
Captures BOTH system audio (what others say) AND your microphone (what you
say) simultaneously. Records them as separate tracks so the transcriber can
tag your voice as "Me" and use diarization to distinguish other speakers.

Usage:
    python capture.py                          # list audio sources
    python capture.py --record                 # record system + mic (dual)
    python capture.py --record --mic 5         # specify mic source index
    python capture.py --record --system-only   # only system audio (no mic)
    python capture.py --record --auto          # auto-transcribe when done

How it works:
    1. Finds your default monitor source (system audio loopback)
    2. Finds your default microphone
    3. Records both simultaneously as separate WAV files
    4. On stop, merges them into a stereo file (L=mic, R=system)
    5. Passes all three files to transcribe.py which uses channel separation
       to know exactly which audio is "Me" vs "Others"

Requirements:
    pip install numpy scipy
    # parec + ffmpeg must be available
    # PipeWire or PulseAudio must be running (default on Fedora)
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Audio source discovery
# ---------------------------------------------------------------------------

def list_pipewire_sources() -> list[dict]:
    """
    List available audio sources using pactl (works with both
    PipeWire and PulseAudio).
    """
    sources = []

    try:
        result = subprocess.run(
            ["pactl", "--format=json", "list", "sources"],
            capture_output=True, text=True, check=True,
        )
        pa_sources = json.loads(result.stdout)

        for src in pa_sources:
            name = src.get("name", "")
            desc = src.get("description", "")
            index = src.get("index", -1)
            state = src.get("state", "unknown")
            is_monitor = ".monitor" in name or "Monitor" in desc

            sources.append({
                "index": index,
                "name": name,
                "description": desc,
                "is_monitor": is_monitor,
                "state": state,
            })

    except FileNotFoundError:
        print("❌ pactl not found. Is PipeWire or PulseAudio installed?")
        print("   On Fedora: sudo dnf install pipewire-pulseaudio pulseaudio-utils")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ pactl failed: {e}")
        sys.exit(1)

    return sources


def find_default_monitor(sources: list[dict]) -> dict | None:
    """Find the monitor source for the default audio output sink."""
    try:
        result = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True, text=True, check=True,
        )
        default_sink = result.stdout.strip()
        monitor_name = f"{default_sink}.monitor"

        for src in sources:
            if src["name"] == monitor_name:
                return src

        # Fallback: any monitor source
        for src in sources:
            if src["is_monitor"]:
                return src

    except Exception as e:
        print(f"⚠️  Could not detect default monitor: {e}")

    return None


def find_default_mic(sources: list[dict]) -> dict | None:
    """Find the default microphone input source."""
    try:
        result = subprocess.run(
            ["pactl", "get-default-source"],
            capture_output=True, text=True, check=True,
        )
        default_source = result.stdout.strip()

        for src in sources:
            if src["name"] == default_source and not src["is_monitor"]:
                return src

        # Fallback: first non-monitor source
        for src in sources:
            if not src["is_monitor"]:
                return src

    except Exception as e:
        print(f"⚠️  Could not detect default mic: {e}")

    return None


def print_sources(sources: list[dict]):
    """Pretty-print available audio sources."""
    print("\n🔊 Available audio sources:")
    print("=" * 70)

    monitors = [s for s in sources if s["is_monitor"]]
    inputs = [s for s in sources if not s["is_monitor"]]

    if monitors:
        print("\n  📡 Monitor sources (captures what others say — system audio):")
        for src in monitors:
            state_icon = "🟢" if src["state"] == "RUNNING" else "⚪"
            print(f"    {state_icon} [{src['index']}] {src['description']}")
            print(f"         {src['name']}")

    if inputs:
        print("\n  🎤 Microphone sources (captures what you say):")
        for src in inputs:
            state_icon = "🟢" if src["state"] == "RUNNING" else "⚪"
            print(f"    {state_icon} [{src['index']}] {src['description']}")
            print(f"         {src['name']}")

    print("=" * 70)

    default_monitor = find_default_monitor(sources)
    default_mic = find_default_mic(sources)

    if default_monitor:
        print(f"\n  📡 Auto-detected system audio: [{default_monitor['index']}] {default_monitor['description']}")
    if default_mic:
        print(f"  🎤 Auto-detected microphone:  [{default_mic['index']}] {default_mic['description']}")

    print()


# ---------------------------------------------------------------------------
# Dual-source recorder (mic + system audio via parec)
# ---------------------------------------------------------------------------

class ParecRecorder:
    """
    Records from a single PipeWire/PulseAudio source using parec → ffmpeg.
    Used internally — DualRecorder runs two of these in parallel.
    """

    def __init__(self, source_name: str, output_path: str,
                 sample_rate: int = 16000, channels: int = 1):
        self.source_name = source_name
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.parec_proc = None
        self.ffmpeg_proc = None

    def start(self):
        parec_cmd = [
            "parec",
            "--device", self.source_name,
            "--format=float32le",
            "--rate", str(self.sample_rate),
            "--channels", str(self.channels),
        ]

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "f32le",
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "-i", "pipe:0",
            self.output_path,
        ]

        self.parec_proc = subprocess.Popen(
            parec_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        self.ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=self.parec_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self):
        if self.parec_proc:
            self.parec_proc.terminate()
            self.parec_proc.wait(timeout=5)
        if self.ffmpeg_proc:
            self.ffmpeg_proc.wait(timeout=10)


class DualRecorder:
    """
    Records from TWO sources simultaneously:
      - System monitor (what others say in the meeting)
      - Microphone (what you say)

    Outputs:
      - {name}_mic.wav      — your voice only
      - {name}_system.wav   — system/meeting audio only
      - {name}_combined.wav — stereo: L=mic, R=system (for playback)
    """

    def __init__(self, monitor_source: str, mic_source: str,
                 output_dir: str, base_name: str,
                 sample_rate: int = 16000):
        self.monitor_source = monitor_source
        self.mic_source = mic_source
        self.output_dir = output_dir
        self.base_name = base_name
        self.sample_rate = sample_rate
        self.start_time = None
        self._stop_event = threading.Event()
        self._monitor_thread = None

        self.mic_path = os.path.join(output_dir, f"{base_name}_mic.wav")
        self.system_path = os.path.join(output_dir, f"{base_name}_system.wav")
        self.combined_path = os.path.join(output_dir, f"{base_name}_combined.wav")

        self.mic_recorder = ParecRecorder(mic_source, self.mic_path, sample_rate)
        self.system_recorder = ParecRecorder(monitor_source, self.system_path, sample_rate)

    def start(self):
        self.start_time = time.time()

        # Start both recorders simultaneously
        self.system_recorder.start()
        self.mic_recorder.start()

        # Duration display
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._print_duration, daemon=True)
        self._monitor_thread.start()

    def _print_duration(self):
        while not self._stop_event.is_set():
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            hours, mins = divmod(mins, 60)
            print(f"\r  ⏺️  Recording... {hours:02d}:{mins:02d}:{secs:02d}  [mic + system]",
                  end="", flush=True)
            self._stop_event.wait(timeout=1.0)

    def stop(self) -> dict:
        """Stop recording and merge tracks. Returns dict of output paths."""
        self._stop_event.set()

        # Stop both recorders
        self.mic_recorder.stop()
        self.system_recorder.stop()

        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        print(f"\r  ⏹️  Recording stopped. Duration: {mins:02d}:{secs:02d}              ")

        # Merge into a stereo file (L=mic, R=system) for combined playback
        print("  🔀 Merging mic + system into stereo file...")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", self.mic_path,
                    "-i", self.system_path,
                    "-filter_complex",
                    "[0:a][1:a]amerge=inputs=2[a]",
                    "-map", "[a]",
                    "-ac", "2",
                    self.combined_path,
                ],
                capture_output=True, check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Merge failed: {e.stderr.decode()[:200]}")
            self.combined_path = None

        return {
            "mic": self.mic_path,
            "system": self.system_path,
            "combined": self.combined_path,
        }


class SingleRecorder:
    """Records from a single source (system-only or mic-only mode)."""

    def __init__(self, source_name: str, output_path: str,
                 sample_rate: int = 16000, label: str = "system"):
        self.label = label
        self.output_path = output_path
        self.recorder = ParecRecorder(source_name, output_path, sample_rate)
        self.start_time = None
        self._stop_event = threading.Event()
        self._monitor_thread = None

    def start(self):
        self.start_time = time.time()
        self.recorder.start()
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._print_duration, daemon=True)
        self._monitor_thread.start()

    def _print_duration(self):
        while not self._stop_event.is_set():
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            hours, mins = divmod(mins, 60)
            print(f"\r  ⏺️  Recording... {hours:02d}:{mins:02d}:{secs:02d}  [{self.label}]",
                  end="", flush=True)
            self._stop_event.wait(timeout=1.0)

    def stop(self) -> str:
        self._stop_event.set()
        self.recorder.stop()
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        print(f"\r  ⏹️  Recording stopped. Duration: {mins:02d}:{secs:02d}              ")
        return self.output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="🎙️ Live Audio Capture — Dual-source (Mic + System Audio)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
How dual-source capture works:
  Your mic is recorded separately from system audio (Teams/Zoom/Meet).
  This lets the transcriber tag your voice as "Me" with 100% accuracy,
  then use diarization to distinguish other speakers on the call.

Examples:
  %(prog)s                                        # list sources
  %(prog)s --record                               # dual capture (mic + system)
  %(prog)s --record --mic 5                       # specify mic by index
  %(prog)s --record --system-only                 # system audio only (no mic)
  %(prog)s --record --auto --diarize --summarize  # full pipeline
  %(prog)s --record --name "weekly-standup"        # custom filename
        """,
    )

    parser.add_argument(
        "--record", "-r",
        action="store_true",
        help="Start recording (Ctrl+C to stop)",
    )
    parser.add_argument(
        "--monitor", "-M",
        type=int, default=None,
        help="Monitor source index for system audio (default: auto-detect)",
    )
    parser.add_argument(
        "--mic",
        type=int, default=None,
        help="Microphone source index (default: auto-detect)",
    )
    parser.add_argument(
        "--system-only",
        action="store_true",
        help="Only capture system audio (no microphone)",
    )
    parser.add_argument(
        "--mic-only",
        action="store_true",
        help="Only capture microphone (no system audio)",
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Custom name for the recording (default: meeting_<timestamp>)",
    )
    _default_output = os.path.expanduser("~/Documents/notes2/meetings")
    parser.add_argument(
        "--output", "-o",
        default=_default_output,
        help=f"Output directory for recordings and notes (default: {_default_output})",
    )
    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        help="Automatically run transcribe.py after recording",
    )
    parser.add_argument(
        "--engine", "-e",
        default="whisper",
        choices=["whisper", "whisper-cpp"],
        help="Transcription engine: whisper (Python/CPU) or whisper-cpp (C++/Vulkan GPU). Default: whisper",
    )
    parser.add_argument(
        "--model", "-m",
        default="turbo",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model for auto-transcription (default: turbo)",
    )
    parser.add_argument(
        "--diarize", "-d",
        action="store_true",
        help="Enable diarization of other speakers in auto-transcription",
    )
    parser.add_argument(
        "--summarize", "-s",
        action="store_true",
        help="Enable summarization in auto-transcription",
    )
    parser.add_argument(
        "--llm",
        default="claude",
        choices=["claude", "ollama", "openai"],
        help="LLM backend for summarization (default: claude — uses your OAuth token)",
    )
    parser.add_argument(
        "--dictionary", "--dict",
        default=None,
        help="Path to vocabulary dictionary file (names, products, etc.) for better recognition",
    )

    args = parser.parse_args()

    # --- Discover sources ---
    sources = list_pipewire_sources()

    if not args.record:
        print_sources(sources)
        print("To start recording (mic + system audio):")
        print("  python capture.py --record")
        print("  python capture.py --record --auto --diarize --summarize")
        return

    # --- Resolve monitor source ---
    monitor_src = None
    if not args.mic_only:
        if args.monitor is not None:
            for s in sources:
                if s["index"] == args.monitor:
                    monitor_src = s
                    break
            if not monitor_src:
                print(f"❌ Monitor source [{args.monitor}] not found.")
                print_sources(sources)
                sys.exit(1)
        else:
            monitor_src = find_default_monitor(sources)
            if not monitor_src:
                print("❌ No monitor source detected.")
                print_sources(sources)
                sys.exit(1)

    # --- Resolve mic source ---
    mic_src = None
    if not args.system_only:
        if args.mic is not None:
            for s in sources:
                if s["index"] == args.mic:
                    mic_src = s
                    break
            if not mic_src:
                print(f"❌ Mic source [{args.mic}] not found.")
                print_sources(sources)
                sys.exit(1)
        else:
            mic_src = find_default_mic(sources)
            if not mic_src:
                print("⚠️  No microphone detected. Falling back to system-only mode.")
                args.system_only = True

    # --- Determine recording mode ---
    dual_mode = not args.system_only and not args.mic_only and monitor_src and mic_src

    # --- Set up output ---
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    base_name = args.name or f"meeting_{timestamp}"

    # --- Print info ---
    print("=" * 60)
    print("🎙️  Live Audio Capture")
    print("=" * 60)
    if dual_mode:
        print(f"  Mode:    DUAL (mic + system)")
        print(f"  🎤 Mic:    [{mic_src['index']}] {mic_src['description']}")
        print(f"  📡 System: [{monitor_src['index']}] {monitor_src['description']}")
    elif args.mic_only:
        print(f"  Mode:    Mic only")
        print(f"  🎤 Mic:    [{mic_src['index']}] {mic_src['description']}")
    else:
        print(f"  Mode:    System only")
        print(f"  📡 System: [{monitor_src['index']}] {monitor_src['description']}")
    print(f"  Output:  {output_dir}/{base_name}_*.wav")
    print(f"  Auto-transcribe: {'yes' if args.auto else 'no'}")
    print("=" * 60)
    print()
    print("  Press Ctrl+C to stop recording.")
    print()

    # --- Start recording ---
    if dual_mode:
        recorder = DualRecorder(
            monitor_source=monitor_src["name"],
            mic_source=mic_src["name"],
            output_dir=output_dir,
            base_name=base_name,
        )
        recorder.start()

        try:
            signal.pause()
        except KeyboardInterrupt:
            pass

        paths = recorder.stop()

        # Check file sizes
        for label, path in paths.items():
            if path and os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  📄 {label:8s}: {path} ({size_mb:.1f} MB)")

        print(f"\n✅ Recording saved!")
        print(f"   Mic (you):     {paths['mic']}")
        print(f"   System (them): {paths['system']}")
        if paths.get("combined"):
            print(f"   Combined:      {paths['combined']}")

        # --- Auto-transcribe ---
        if args.auto:
            _run_transcription(args, output_dir, paths)
        else:
            print(f"\nTo transcribe:")
            print(f"  python transcribe.py --mic {paths['mic']} --system {paths['system']}")
            print(f"  python transcribe.py --mic {paths['mic']} --system {paths['system']} --diarize --summarize")

    else:
        # Single-source mode
        if args.mic_only:
            src_name = mic_src["name"]
            label = "mic"
        else:
            src_name = monitor_src["name"]
            label = "system"

        wav_path = os.path.join(output_dir, f"{base_name}_{label}.wav")
        recorder = SingleRecorder(src_name, wav_path, label=label)
        recorder.start()

        try:
            signal.pause()
        except KeyboardInterrupt:
            pass

        recorder.stop()

        size_mb = os.path.getsize(wav_path) / (1024 * 1024)
        print(f"\n✅ Saved: {wav_path} ({size_mb:.1f} MB)")

        if args.auto:
            paths = {"mic": None, "system": None}
            paths[label] = wav_path
            _run_transcription(args, output_dir, paths)
        else:
            print(f"\nTo transcribe:")
            print(f"  python transcribe.py {wav_path}")


def _run_transcription(args, output_dir: str, paths: dict):
    """Run transcribe.py with the appropriate arguments."""
    print(f"\n🚀 Starting transcription...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    transcribe_script = os.path.join(script_dir, "transcribe.py")

    cmd = [sys.executable, transcribe_script]

    # Pass dual-channel paths if available
    if paths.get("mic") and paths.get("system"):
        cmd.extend(["--mic", paths["mic"], "--system", paths["system"]])
    elif paths.get("mic"):
        cmd.extend([paths["mic"]])
    elif paths.get("system"):
        cmd.extend([paths["system"]])

    cmd.extend(["--engine", args.engine])
    cmd.extend(["--model", args.model])

    if args.diarize:
        cmd.append("--diarize")
    if args.summarize:
        cmd.append("--summarize")
        cmd.extend(["--llm", args.llm])

    cmd.extend(["--output", output_dir])

    if hasattr(args, "dictionary") and args.dictionary:
        cmd.extend(["--dictionary", args.dictionary])

    print(f"  Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
