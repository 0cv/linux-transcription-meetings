#!/usr/bin/env python3
"""
Meeting Transcriber CLI
=======================
Transcribes meeting audio using OpenAI Whisper, identifies speakers,
and generates Obsidian-ready meeting notes.

Supports two modes:
  1. Single file:    python transcribe.py meeting.wav
  2. Dual-channel:   python transcribe.py --mic mic.wav --system system.wav

Dual-channel mode is the recommended approach for live meetings:
  - Your mic track is tagged as "Me" automatically
  - System audio (others) is transcribed and optionally diarized
  - Result: a transcript where you are always "Me" and others are
    "Speaker 1", "Speaker 2", etc.

Requirements:
    pip install openai-whisper
    pip install pyannote.audio          # for --diarize
    pip install ollama                   # for --summarize with local LLM
    # OR set OPENAI_API_KEY env var      # for --summarize with OpenAI
"""

import argparse
import json
import os
import re
import sys
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dictionary(dict_path: str | None = None) -> list[str]:
    """Load vocabulary terms from a dictionary file.

    Returns a list of terms (one per line, comments and blanks stripped).
    If no path given, looks for dictionary.txt next to this script.
    """
    if dict_path is None:
        default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionary.txt")
        if os.path.isfile(default):
            dict_path = default
        else:
            return []

    if not os.path.isfile(dict_path):
        print(f"⚠️  Dictionary file not found: {dict_path}")
        return []

    terms = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)
    if terms:
        print(f"📖 Loaded {len(terms)} vocabulary terms from {dict_path}")
    return terms


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def ensure_wav(audio_path: str) -> str:
    """Convert audio to 16kHz mono WAV if needed (required by pyannote)."""
    if audio_path.lower().endswith(".wav"):
        return audio_path
    tmp = tempfile.mktemp(suffix=".wav")
    print(f"  Converting to WAV for diarization...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", tmp],
        capture_output=True,
        check=True,
    )
    return tmp


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, model_name: str = "turbo",
                     language: str | None = None,
                     task: str = "transcribe",
                     engine: str = "whisper",
                     diarize: bool = False,
                     dictionary_terms: list[str] | None = None) -> dict:
    """Run Whisper transcription and return the result dict.

    engine: "whisper" (Python, CPU) or "whisper-cpp" (C++, Vulkan GPU).
    diarize: When True and engine is whisper-cpp, enables -tdrz speaker turns.
    dictionary_terms: List of vocabulary terms to bias the decoder toward.
    """
    if engine == "whisper-cpp":
        return transcribe_audio_cpp(audio_path, model_name, language, task, diarize, dictionary_terms)

    import whisper

    print(f"\n📝 Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    print(f"📝 Transcribing: {os.path.basename(audio_path)}")
    options = {"task": task, "verbose": False}
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)
    print(f"✅ Transcription complete  ({len(result['segments'])} segments)")
    return result


# ---------------------------------------------------------------------------
# whisper.cpp backend (Vulkan GPU-accelerated)
# ---------------------------------------------------------------------------

# Model name mapping: OpenAI Python names → ggml model file names
_CPP_MODEL_MAP = {
    "tiny":   "ggml-tiny.bin",
    "base":   "ggml-base.bin",
    "small":  "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large":  "ggml-large-v3.bin",
    "turbo":  "ggml-large-v3-turbo.bin",
}


def _find_whisper_cpp() -> tuple[str, str]:
    """Locate the whisper-cli binary and models directory.

    Search order:
      1. WHISPER_CPP_DIR env var  (e.g. ~/whisper.cpp)
      2. ~/whisper.cpp
      3. whisper-cli on PATH
    Returns (cli_path, models_dir).
    """
    # Check env var or default location
    cpp_dir = os.environ.get("WHISPER_CPP_DIR", os.path.expanduser("~/whisper.cpp"))

    cli_candidates = [
        os.path.join(cpp_dir, "build", "bin", "whisper-cli"),
        os.path.join(cpp_dir, "build", "bin", "Release", "whisper-cli"),
        os.path.join(cpp_dir, "main"),  # older builds
    ]

    for candidate in cli_candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            models_dir = os.path.join(cpp_dir, "models")
            return candidate, models_dir

    # Fall back to PATH
    import shutil
    path_cli = shutil.which("whisper-cli")
    if path_cli:
        # Guess models dir relative to the binary
        bin_dir = os.path.dirname(path_cli)
        possible_models = os.path.join(os.path.dirname(bin_dir), "models")
        if os.path.isdir(possible_models):
            return path_cli, possible_models
        return path_cli, os.path.expanduser("~/whisper.cpp/models")

    raise FileNotFoundError(
        "whisper-cli not found. Build whisper.cpp with Vulkan support:\n"
        "  cd ~ && git clone https://github.com/ggml-org/whisper.cpp\n"
        "  cd whisper.cpp && cmake -B build -DGGML_VULKAN=1 && cmake --build build -j\n"
        "  bash models/download-ggml-model.sh large-v3-turbo"
    )


def _ensure_cpp_model(models_dir: str, model_name: str) -> str:
    """Return path to ggml model file, downloading it if necessary."""
    filename = _CPP_MODEL_MAP.get(model_name)
    if not filename:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_CPP_MODEL_MAP.keys())}")

    model_path = os.path.join(models_dir, filename)
    if os.path.isfile(model_path):
        return model_path

    # Try downloading via the bundled script
    download_script = os.path.join(os.path.dirname(models_dir), "models", "download-ggml-model.sh")
    if not os.path.isfile(download_script):
        download_script = os.path.join(os.path.dirname(models_dir), "download-ggml-model.sh")

    # Map our names to the script's expected names
    script_name = model_name
    if model_name == "large":
        script_name = "large-v3"
    elif model_name == "turbo":
        script_name = "large-v3-turbo"

    if os.path.isfile(download_script):
        print(f"⬇️  Downloading ggml model: {script_name}...")
        subprocess.run(["bash", download_script, script_name], cwd=os.path.dirname(download_script), check=True)
        if os.path.isfile(model_path):
            return model_path

    raise FileNotFoundError(
        f"Model file not found: {model_path}\n"
        f"Download it with:  bash models/download-ggml-model.sh {script_name}"
    )


def transcribe_audio_cpp(audio_path: str, model_name: str = "turbo",
                         language: str | None = None,
                         task: str = "transcribe",
                         diarize: bool = False,
                         dictionary_terms: list[str] | None = None) -> dict:
    """Transcribe using whisper.cpp CLI with Vulkan GPU acceleration.

    When diarize=True, passes -tdrz which inserts [SPEAKER_TURN] tokens
    into the text. The returned segments will have "speaker_turn" markers.
    dictionary_terms: If provided, joined into a --prompt string to bias
    the decoder toward recognizing these words (names, products, etc.).

    Returns a dict compatible with the Python whisper output format:
    {"segments": [{"start": ..., "end": ..., "text": ...}], "language": ...}
    """
    cli_path, models_dir = _find_whisper_cpp()
    model_path = _ensure_cpp_model(models_dir, model_name)

    print(f"\n📝 whisper.cpp ({model_name}) with Vulkan GPU")
    print(f"   Binary: {cli_path}")
    print(f"   Model:  {model_path}")
    if diarize:
        print(f"   Speaker turn detection: enabled (-tdrz)")

    # whisper.cpp requires 16-bit 16kHz mono WAV
    wav_path = ensure_wav(audio_path)

    # Build command
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_out = tmp.name

    cmd = [
        cli_path,
        "-m", model_path,
        "-f", wav_path,
        "--output-json-full",
        "-oj",                 # output JSON
        "-of", json_out.replace(".json", ""),  # output file prefix (whisper-cli appends .json)
        "-t", str(min(os.cpu_count() or 4, 8)),  # threads for CPU-side work
    ]

    if diarize:
        cmd.append("-tdrz")    # enable tinydiarize speaker turn detection

    if language:
        cmd.extend(["-l", language])
    if task == "translate":
        cmd.append("--translate")

    # Feed vocabulary terms as initial prompt to bias the decoder
    if dictionary_terms:
        prompt_text = ", ".join(dictionary_terms)
        cmd.extend(["--prompt", prompt_text])
        print(f"   Vocabulary prompt: {len(dictionary_terms)} terms loaded")

    print(f"📝 Transcribing: {os.path.basename(audio_path)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ whisper.cpp failed:\n{e.stderr}")
        raise

    # Parse JSON output
    actual_json = json_out.replace(".json", "") + ".json"
    if not os.path.isfile(actual_json):
        # whisper-cli may name it differently
        for candidate in [json_out, json_out + ".json"]:
            if os.path.isfile(candidate):
                actual_json = candidate
                break

    with open(actual_json, "r", encoding="utf-8") as f:
        cpp_result = json.load(f)

    # Clean up temp file
    try:
        os.unlink(actual_json)
    except OSError:
        pass

    # Convert whisper.cpp JSON → Python whisper format
    raw_segments = []
    transcription = cpp_result.get("transcription", cpp_result.get("segments", []))
    for seg in transcription:
        # whisper.cpp uses "timestamps" dict or "t0"/"t1" fields
        if "timestamps" in seg:
            start_ms = seg["timestamps"].get("from", "00:00:00,000")
            end_ms = seg["timestamps"].get("to", "00:00:00,000")
            start = _parse_cpp_timestamp(start_ms)
            end = _parse_cpp_timestamp(end_ms)
        elif "offsets" in seg:
            start = seg["offsets"].get("from", 0) / 1000.0
            end = seg["offsets"].get("to", 0) / 1000.0
        else:
            start = seg.get("start", seg.get("t0", 0))
            end = seg.get("end", seg.get("t1", 0))
            # If in milliseconds, convert
            if isinstance(start, int) and start > 100000:
                start = start / 1000.0
                end = end / 1000.0

        text = seg.get("text", "").strip()
        if text:
            raw_segments.append({"start": start, "end": end, "text": text})

    # If diarize was enabled, split segments on [SPEAKER_TURN] markers
    # and assign speaker labels
    if diarize:
        segments = _split_speaker_turns(raw_segments)
    else:
        segments = raw_segments

    detected_lang = cpp_result.get("result", {}).get("language", language or "unknown")
    if detected_lang == "unknown" and language:
        detected_lang = language

    print(f"✅ Transcription complete  ({len(segments)} segments)")
    return {"segments": segments, "language": detected_lang}


def _split_speaker_turns(segments: list[dict]) -> list[dict]:
    """Process [SPEAKER_TURN] markers from whisper.cpp -tdrz output.

    Splits segments at [SPEAKER_TURN] boundaries and assigns incrementing
    speaker labels (Speaker 1, Speaker 2, ...). A new [SPEAKER_TURN] marker
    means a different person is now speaking.
    """
    import re
    TURN_MARKER = "[SPEAKER_TURN]"

    result = []
    speaker_num = 1

    for seg in segments:
        text = seg["text"]

        # Check if segment contains speaker turn markers
        if TURN_MARKER in text:
            # Split on the marker
            parts = text.split(TURN_MARKER)
            seg_duration = seg["end"] - seg["start"]
            total_chars = max(sum(len(p.strip()) for p in parts), 1)
            time_cursor = seg["start"]

            for i, part in enumerate(parts):
                part = part.strip()
                if i > 0:
                    # Each [SPEAKER_TURN] means a new speaker
                    speaker_num += 1

                if part:
                    # Approximate sub-timing proportional to text length
                    frac = len(part) / total_chars
                    part_duration = seg_duration * frac
                    result.append({
                        "start": time_cursor,
                        "end": time_cursor + part_duration,
                        "text": part,
                        "speaker": f"Speaker {speaker_num}",
                    })
                    time_cursor += part_duration
        else:
            # No turn marker — same speaker continues
            result.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "speaker": f"Speaker {speaker_num}",
            })

    # Renumber speakers sequentially (collapse gaps)
    seen = {}
    counter = 1
    for seg in result:
        raw = seg["speaker"]
        if raw not in seen:
            seen[raw] = f"Speaker {counter}"
            counter += 1
        seg["speaker"] = seen[raw]

    unique = len(seen)
    print(f"   Speaker turns detected: {unique} distinct speakers")
    return result


def _parse_cpp_timestamp(ts: str) -> float:
    """Parse whisper.cpp timestamp like '00:01:23,456' → seconds."""
    ts = ts.strip()
    if "," in ts:
        time_part, ms_part = ts.rsplit(",", 1)
    elif "." in ts:
        time_part, ms_part = ts.rsplit(".", 1)
    else:
        time_part = ts
        ms_part = "0"
    parts = time_part.split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h * 3600 + m * 60 + s + int(ms_part) / 1000.0


# ---------------------------------------------------------------------------
# Dual-channel transcription
# ---------------------------------------------------------------------------

def transcribe_dual_channel(mic_path: str, system_path: str,
                            model_name: str = "turbo",
                            language: str | None = None,
                            task: str = "transcribe",
                            diarize: bool = False,
                            hf_token: str | None = None,
                            engine: str = "whisper",
                            dictionary_terms: list[str] | None = None) -> list[dict]:
    """
    Transcribe mic and system audio separately, then merge timelines.

    - Mic segments are tagged as "Me"
    - System segments are tagged as "Speaker 1", "Speaker 2", etc.
      (via pyannote diarization if enabled, otherwise all "Others")

    engine: "whisper" (Python, CPU) or "whisper-cpp" (C++, Vulkan GPU).
    Returns a unified list of segments sorted by start time.
    """
    # --- Transcribe mic (you) ---
    print(f"\n🎤 Transcribing your mic...")
    mic_result = transcribe_audio(mic_path, model_name, language, task, engine,
                                  dictionary_terms=dictionary_terms)
    mic_segments = mic_result.get("segments", [])
    print(f"   {len(mic_segments)} segments from your mic")

    detected_lang = mic_result.get("language", "unknown")

    # --- Transcribe system audio (others) ---
    # When diarizing with whisper-cpp, pass diarize=True so -tdrz is used
    # and segments come back pre-tagged with speaker labels.
    use_cpp_diarize = diarize and engine == "whisper-cpp"

    print(f"\n📡 Transcribing system audio (others)...")
    system_result = transcribe_audio(
        system_path, model_name, language, task, engine,
        diarize=use_cpp_diarize,
        dictionary_terms=dictionary_terms,
    )
    system_segments = system_result.get("segments", [])
    print(f"   {len(system_segments)} segments from system audio")

    if not detected_lang or detected_lang == "unknown":
        detected_lang = system_result.get("language", "unknown")

    # --- Tag mic segments as "Me" ---
    me_segments = []
    for seg in mic_segments:
        text = seg["text"].strip()
        if not text:
            continue
        me_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": "Me",
            "text": text,
        })

    # --- Tag system segments ---
    if use_cpp_diarize and system_segments:
        # whisper.cpp -tdrz already tagged segments with "Speaker N" labels
        other_segments = [
            seg for seg in system_segments
            if seg.get("text", "").strip()
        ]

    elif diarize and engine == "whisper" and system_segments:
        # Fall back to pyannote for Python whisper engine
        print(f"\n🎙️  Diarizing other speakers (pyannote)...")
        diarization_segments = diarize_audio(system_path, hf_token)

        other_segments = []
        for seg in system_segments:
            text = seg["text"].strip()
            if not text:
                continue

            # Find best matching speaker from diarization
            best_speaker = "Others"
            best_overlap = 0

            for ds in diarization_segments:
                overlap_start = max(seg["start"], ds["start"])
                overlap_end = min(seg["end"], ds["end"])
                overlap = max(0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = ds["speaker"]

            other_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": best_speaker,
                "text": text,
            })

        # Rename speakers to friendly names: SPEAKER_00 → "Speaker 1", etc.
        speaker_map = {}
        speaker_counter = 1
        for seg in other_segments:
            raw = seg["speaker"]
            if raw not in speaker_map and raw != "Others":
                speaker_map[raw] = f"Speaker {speaker_counter}"
                speaker_counter += 1
            seg["speaker"] = speaker_map.get(raw, "Others")

    else:
        # No diarization — all system audio is "Others"
        other_segments = []
        for seg in system_segments:
            text = seg["text"].strip()
            if not text:
                continue
            other_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": "Others",
                "text": text,
            })

    # --- Merge and sort by time ---
    all_segments = me_segments + other_segments
    all_segments.sort(key=lambda s: s["start"])

    # --- Remove duplicate/overlapping segments ---
    # When you speak, your voice may bleed into the system audio via echo.
    # Remove system segments that overlap heavily with mic segments.
    cleaned = _remove_echo_duplicates(all_segments)

    num_speakers = len(set(s["speaker"] for s in cleaned))
    print(f"\n✅ Merged timeline: {len(cleaned)} segments, {num_speakers} speakers")

    return cleaned, detected_lang


def _remove_echo_duplicates(segments: list[dict]) -> list[dict]:
    """
    Remove likely echo/duplicate segments where the system audio picks up
    your voice through the speakers. If a "Me" segment and an "Others"
    segment overlap significantly and have similar text, drop the "Others" one.
    """
    me_segs = [s for s in segments if s["speaker"] == "Me"]
    other_segs = [s for s in segments if s["speaker"] != "Me"]

    keep = []
    for oseg in other_segs:
        is_echo = False
        for mseg in me_segs:
            # Check time overlap
            overlap_start = max(oseg["start"], mseg["start"])
            overlap_end = min(oseg["end"], mseg["end"])
            overlap = max(0, overlap_end - overlap_start)
            oseg_dur = oseg["end"] - oseg["start"]

            if oseg_dur > 0 and overlap / oseg_dur > 0.5:
                # Significant time overlap — check text similarity
                o_words = set(oseg["text"].lower().split())
                m_words = set(mseg["text"].lower().split())
                if o_words and m_words:
                    common = len(o_words & m_words)
                    similarity = common / max(len(o_words), len(m_words))
                    if similarity > 0.4:
                        is_echo = True
                        break

        if not is_echo:
            keep.append(oseg)

    result = me_segs + keep
    result.sort(key=lambda s: s["start"])
    return result


# ---------------------------------------------------------------------------
# Speaker Diarization
# ---------------------------------------------------------------------------

def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """
    Run pyannote speaker diarization.
    Returns list of {start, end, speaker} dicts.
    """
    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️  No HuggingFace token found. Set HF_TOKEN env var or pass --hf-token.")
        print("   You need access to pyannote/speaker-diarization-3.1")
        print("   Get a token at https://huggingface.co/settings/tokens")
        sys.exit(1)

    wav_path = ensure_wav(audio_path)

    print(f"  Running pyannote diarization...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )

    diarization = pipeline(wav_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    # Clean up temp WAV
    if wav_path != audio_path and os.path.exists(wav_path):
        os.remove(wav_path)

    unique_speakers = len(set(s["speaker"] for s in segments))
    print(f"  Found {unique_speakers} distinct speakers")
    return segments


def merge_transcript_with_speakers(whisper_segments: list[dict],
                                    diarization_segments: list[dict]) -> list[dict]:
    """
    Assign a speaker label to each Whisper segment based on overlap
    with diarization segments. Used for single-file mode.
    """
    merged = []
    for ws in whisper_segments:
        best_speaker = "Unknown"
        best_overlap = 0

        for ds in diarization_segments:
            overlap_start = max(ws["start"], ds["start"])
            overlap_end = min(ws["end"], ds["end"])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = ds["speaker"]

        merged.append({
            "start": ws["start"],
            "end": ws["end"],
            "speaker": best_speaker,
            "text": ws["text"].strip(),
        })

    # Rename to friendly names
    speaker_map = {}
    counter = 1
    for seg in merged:
        raw = seg["speaker"]
        if raw not in speaker_map and raw != "Unknown":
            speaker_map[raw] = f"Speaker {counter}"
            counter += 1
        seg["speaker"] = speaker_map.get(raw, "Unknown")

    return merged


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def summarize_with_ollama(transcript_text: str, model: str = "qwen3:8b") -> str:
    """Summarize using a local Ollama model (default: Qwen 3 8B)."""
    try:
        import ollama
    except ImportError:
        print("⚠️  ollama package not installed. Run: pip install ollama")
        return ""

    print(f"\n🤖 Summarizing with Ollama ({model})...")
    prompt = _build_summary_prompt(transcript_text)

    try:
        response = ollama.chat(model=model, messages=[
            {"role": "user", "content": prompt}
        ])
        return response["message"]["content"]
    except Exception as e:
        print(f"⚠️  Ollama summarization failed: {e}")
        print(f"   Make sure the model is pulled: ollama pull {model}")
        return ""


def _find_claude_auth_method() -> str:
    """
    Determine the best Claude authentication method.

    Returns one of:
      "oauth"   — CLAUDE_CODE_OAUTH_TOKEN is set → use `claude` CLI
      "apikey"  — ANTHROPIC_API_KEY is set → use anthropic Python SDK
      ""        — nothing found
    """
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return "oauth"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "apikey"
    # Check if claude CLI is available and authenticated
    import shutil
    if shutil.which("claude"):
        # claude CLI can use its own stored credentials
        return "oauth"
    return ""


def _summarize_via_claude_cli(prompt: str, model: str) -> str:
    """Call the `claude` CLI in pipe mode (-p) to summarize.

    This is the supported way to use subscription OAuth tokens.
    The claude CLI reads CLAUDE_CODE_OAUTH_TOKEN and handles auth internally.
    """
    import shutil

    claude_bin = shutil.which("claude")
    if not claude_bin:
        print("⚠️  claude CLI not found on PATH.")
        print("   Install it: https://docs.claude.com/en/docs/claude-code")
        return ""

    cmd = [claude_bin, "-p", "--model", model]

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"⚠️  claude CLI failed (exit {result.returncode}):")
            print(f"   {result.stderr.strip()}")
            return ""
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("⚠️  Claude CLI timed out after 120s.")
        return ""
    except Exception as e:
        print(f"⚠️  Claude CLI error: {e}")
        return ""


def _summarize_via_sdk(prompt: str, model: str) -> str:
    """Call Claude via the anthropic Python SDK with ANTHROPIC_API_KEY."""
    try:
        import anthropic
    except ImportError:
        print("⚠️  anthropic package not installed. Run: pip install anthropic")
        return ""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        print(f"⚠️  Claude API call failed: {e}")
        return ""


def summarize_with_claude(transcript_text: str,
                          model: str = "claude-opus-4-20250514",
                          dictionary_terms: list[str] | None = None) -> str:
    """
    Summarize using Claude.

    Authentication (in order of precedence):
      1. CLAUDE_CODE_OAUTH_TOKEN env var → pipes through `claude -p` CLI
         Set it once:  export CLAUDE_CODE_OAUTH_TOKEN=$(claude setup-token)
         Or:           export CLAUDE_CODE_OAUTH_TOKEN=$(claude auth print-token)
      2. ANTHROPIC_API_KEY env var → uses anthropic Python SDK directly
      3. `claude` CLI on PATH → uses whatever auth claude has stored
    """
    auth = _find_claude_auth_method()

    if not auth:
        print("⚠️  No Claude credentials found.")
        print("   Option A (recommended — uses your subscription):")
        print("     claude setup-token")
        print("     export CLAUDE_CODE_OAUTH_TOKEN=<token from above>")
        print("   Option B (Console API key):")
        print("     export ANTHROPIC_API_KEY=sk-ant-...")
        return ""

    print(f"\n🤖 Summarizing with Claude ({model})...")
    prompt = _build_summary_prompt(transcript_text, dictionary_terms)

    if auth == "oauth":
        return _summarize_via_claude_cli(prompt, model)
    else:
        return _summarize_via_sdk(prompt, model)


def summarize_with_openai(transcript_text: str, model: str = "gpt-4o-mini") -> str:
    """Summarize using OpenAI API (fallback option)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping summarization.")
        return ""

    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️  openai package not installed. Run: pip install openai")
        return ""

    print(f"\n🤖 Summarizing with OpenAI ({model})...")
    client = OpenAI(api_key=api_key)
    prompt = _build_summary_prompt(transcript_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️  OpenAI summarization failed: {e}")
        return ""



def _build_summary_prompt(transcript_text: str,
                          dictionary_terms: list[str] | None = None) -> str:
    glossary_block = ""
    if dictionary_terms:
        terms_list = ", ".join(dictionary_terms)
        glossary_block = f"""
IMPORTANT — VOCABULARY GLOSSARY:
The following are correct spellings of names, products, companies, and technical
terms mentioned in this meeting. Use these exact spellings in your summary and
fix any misspellings or phonetic errors in the transcript:
{terms_list}

"""

    return f"""You are a meeting notes assistant. Summarize the following meeting transcript
into well-structured meeting notes. The speaker labeled "Me" is the person who
requested this summary. Use this format:

## Key Topics Discussed
- Bullet points of main topics

## Decisions Made
- Any decisions or agreements reached

## Action Items
- [ ] Action item (Owner — use "Me" for the requester's items)

## Summary
A concise 2-3 paragraph summary of the meeting.
{glossary_block}
---

TRANSCRIPT:
{transcript_text}
"""


# ---------------------------------------------------------------------------
# Output Formatters
# ---------------------------------------------------------------------------

def format_plain_transcript(segments: list[dict], has_speakers: bool) -> str:
    """Format transcript as readable text with speaker labels."""
    lines = []
    current_speaker = None

    for seg in segments:
        ts = seconds_to_timestamp(seg["start"])
        if has_speakers:
            speaker = seg.get("speaker", "Unknown")
            if speaker != current_speaker:
                current_speaker = speaker
                lines.append(f"\n[{ts}] **{speaker}**:")
            lines.append(f"  {seg['text']}")
        else:
            lines.append(f"[{ts}] {seg['text']}")

    return "\n".join(lines)


def generate_obsidian_note(source_label: str, transcript_text: str,
                           summary: str = "", metadata: dict = None) -> str:
    """Generate an Obsidian-compatible markdown meeting note."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    # Clean up the name for a title
    title = source_label.replace("_", " ").replace("-", " ").title()

    note = f"""---
date: {date_str}
time: {time_str}
type: meeting-note
source: {source_label}
model: {metadata.get('model', 'turbo') if metadata else 'turbo'}
language: {metadata.get('language', 'auto') if metadata else 'auto'}
mode: {metadata.get('mode', 'single') if metadata else 'single'}
tags:
  - meeting
  - transcript
---

# {title}

📅 **Date:** {date_str}
⏰ **Time:** {time_str}
🎙️ **Source:** `{source_label}`

"""

    if summary:
        note += f"""---

## Meeting Notes

{summary}

"""

    note += f"""---

## Full Transcript

{transcript_text}
"""

    return note


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="🎙️ Meeting Transcriber — Whisper + Speaker ID + Obsidian Notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Single-file mode (basic):
  %(prog)s meeting.wav
  %(prog)s meeting.mp4 --diarize --summarize

Dual-channel mode (recommended for live meetings):
  %(prog)s --mic mic.wav --system system.wav
  %(prog)s --mic mic.wav --system system.wav --diarize --summarize

In dual-channel mode, your mic is tagged as "Me" and system audio is
diarized to identify other speakers as "Speaker 1", "Speaker 2", etc.
        """,
    )

    # Input: either a single audio file, or --mic + --system
    parser.add_argument(
        "audio", nargs="?", default=None,
        help="Path to audio/video file (single-file mode)",
    )
    parser.add_argument(
        "--mic",
        default=None,
        help="Path to mic recording — your voice (dual-channel mode)",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Path to system audio recording — others (dual-channel mode)",
    )

    # Whisper options
    parser.add_argument(
        "--engine", "-e",
        default="whisper",
        choices=["whisper", "whisper-cpp"],
        help="Transcription engine: whisper (Python, CPU) or whisper-cpp (C++, Vulkan GPU). Default: whisper",
    )
    parser.add_argument(
        "--model", "-m",
        default="turbo",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model (default: turbo)",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Language code (e.g., en, nl, de). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper task (default: transcribe)",
    )

    # Processing options
    parser.add_argument(
        "--diarize", "-d",
        action="store_true",
        help="Enable speaker diarization (pyannote.audio + HF_TOKEN)",
    )
    parser.add_argument(
        "--summarize", "-s",
        action="store_true",
        help="Generate meeting summary using an LLM",
    )
    parser.add_argument(
        "--llm",
        default="claude",
        choices=["claude", "ollama", "openai"],
        help="LLM backend for summarization (default: claude — uses your OAuth token)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model name (default: claude-opus-4-20250514 / qwen3:8b / gpt-4o-mini)",
    )

    # Output options
    _default_output = os.path.expanduser("~/Documents/notes2/meetings")
    parser.add_argument(
        "--output", "-o",
        default=_default_output,
        help=f"Output directory (default: {_default_output})",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for pyannote (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also save raw transcript as JSON",
    )
    parser.add_argument(
        "--dictionary", "--dict",
        default=None,
        help="Path to vocabulary dictionary file (default: dictionary.txt next to this script). "
             "Terms are fed to whisper as --prompt and included in the LLM summary prompt.",
    )

    args = parser.parse_args()

    # --- Load vocabulary dictionary ---
    dictionary_terms = load_dictionary(args.dictionary)

    # --- Determine mode ---
    dual_mode = args.mic is not None or args.system is not None
    single_mode = args.audio is not None

    if not dual_mode and not single_mode:
        parser.print_help()
        sys.exit(1)

    if dual_mode and single_mode:
        print("❌ Use either a single audio file OR --mic/--system, not both.")
        sys.exit(1)

    # =====================================================================
    # DUAL-CHANNEL MODE
    # =====================================================================
    if dual_mode:
        if not args.mic or not args.system:
            print("❌ Dual-channel mode requires both --mic and --system.")
            sys.exit(1)

        mic_path = os.path.abspath(args.mic)
        system_path = os.path.abspath(args.system)

        for p in [mic_path, system_path]:
            if not os.path.isfile(p):
                print(f"❌ File not found: {p}")
                sys.exit(1)

        # Derive output dir and base name
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)

        # Extract base name from system file (remove _system suffix)
        base = Path(system_path).stem
        if base.endswith("_system"):
            base = base[:-7]
        # Avoid double date prefix if base already starts with YYYY-MM-DD
        if re.match(r"\d{4}-\d{2}-\d{2}", base):
            date_prefix = ""
        else:
            date_prefix = datetime.now().strftime("%Y-%m-%d")

        print("=" * 60)
        print("🎙️  Meeting Transcriber — Dual-Channel Mode")
        print("=" * 60)
        engine_label = "whisper.cpp (Vulkan GPU)" if args.engine == "whisper-cpp" else "whisper (Python/CPU)"
        print(f"  🎤 Mic (you):     {mic_path}")
        print(f"  📡 System (them): {system_path}")
        print(f"  Engine:           {engine_label}")
        print(f"  Model:            {args.model}")
        print(f"  Language:         {args.language or 'auto-detect'}")
        print(f"  Diarize others:   {'yes' if args.diarize else 'no'}")
        print(f"  Summarize:        {'yes' if args.summarize else 'no'}")
        print(f"  Output:           {output_dir}")
        print("=" * 60)

        segments, detected_lang = transcribe_dual_channel(
            mic_path=mic_path,
            system_path=system_path,
            model_name=args.model,
            language=args.language,
            task=args.task,
            diarize=args.diarize,
            hf_token=args.hf_token,
            engine=args.engine,
            dictionary_terms=dictionary_terms,
        )

        has_speakers = True
        source_label = base

    # =====================================================================
    # SINGLE-FILE MODE
    # =====================================================================
    else:
        audio_path = os.path.abspath(args.audio)
        if not os.path.isfile(audio_path):
            print(f"❌ File not found: {audio_path}")
            sys.exit(1)

        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)

        base = Path(audio_path).stem
        if re.match(r"\d{4}-\d{2}-\d{2}", base):
            date_prefix = ""
        else:
            date_prefix = datetime.now().strftime("%Y-%m-%d")

        print("=" * 60)
        print("🎙️  Meeting Transcriber — Single-File Mode")
        print("=" * 60)
        engine_label = "whisper.cpp (Vulkan GPU)" if args.engine == "whisper-cpp" else "whisper (Python/CPU)"
        print(f"  Audio:    {audio_path}")
        print(f"  Engine:   {engine_label}")
        print(f"  Model:    {args.model}")
        print(f"  Language: {args.language or 'auto-detect'}")
        print(f"  Diarize:  {'yes' if args.diarize else 'no'}")
        print(f"  Summarize:{'yes' if args.summarize else 'no'}")
        print(f"  Output:   {output_dir}")
        print("=" * 60)

        # For whisper-cpp + diarize, pass diarize=True to get -tdrz speaker turns
        use_cpp_diarize = args.diarize and args.engine == "whisper-cpp"
        result = transcribe_audio(
            audio_path, args.model, args.language, args.task, args.engine,
            diarize=use_cpp_diarize,
            dictionary_terms=dictionary_terms,
        )
        detected_lang = result.get("language", "unknown")
        print(f"  Detected language: {detected_lang}")

        raw_segments = result["segments"]

        has_speakers = False
        if use_cpp_diarize:
            # whisper.cpp -tdrz already tagged segments with speaker labels
            segments = raw_segments
            has_speakers = True
        elif args.diarize and args.engine == "whisper":
            # Fall back to pyannote for Python whisper engine
            diarization_segments = diarize_audio(audio_path, args.hf_token)
            segments = merge_transcript_with_speakers(
                [{"start": s["start"], "end": s["end"], "text": s["text"]}
                 for s in raw_segments],
                diarization_segments,
            )
            has_speakers = True
        else:
            segments = [
                {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in raw_segments
            ]

        source_label = os.path.basename(audio_path)

    # =====================================================================
    # COMMON OUTPUT PIPELINE
    # =====================================================================

    # --- Format transcript ---
    transcript_text = format_plain_transcript(segments, has_speakers)

    # --- Summarize (optional) ---
    summary = ""
    if args.summarize:
        if args.llm == "claude":
            llm_model = args.llm_model or "claude-opus-4-20250514"
            summary = summarize_with_claude(transcript_text, llm_model, dictionary_terms)
        elif args.llm == "ollama":
            llm_model = args.llm_model or "qwen3:8b"
            summary = summarize_with_ollama(transcript_text, llm_model)
        elif args.llm == "openai":
            llm_model = args.llm_model or "gpt-4o-mini"
            summary = summarize_with_openai(transcript_text, llm_model)

    # --- Generate Obsidian note ---
    metadata = {
        "model": args.model,
        "language": detected_lang,
        "mode": "dual-channel" if dual_mode else "single-file",
    }
    obsidian_note = generate_obsidian_note(source_label, transcript_text, summary, metadata)

    # --- Write outputs ---
    prefix = f"{date_prefix}_" if date_prefix else ""
    note_filename = f"{prefix}{base}.md"
    note_path = os.path.join(output_dir, note_filename)
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(obsidian_note)
    print(f"\n📄 Obsidian note saved: {note_path}")

    txt_filename = f"{prefix}{base}_transcript.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    print(f"📄 Transcript saved:   {txt_path}")

    if args.json:
        json_filename = f"{prefix}{base}_transcript.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        print(f"📄 JSON saved:         {json_path}")

    print(f"\n✅ Done! Open {note_path} in Obsidian.")


if __name__ == "__main__":
    main()
