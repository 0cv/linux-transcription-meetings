#!/usr/bin/env python3
"""
Meeting Name Detection
=======================
Detects the current meeting name by scanning window titles for active
Teams/Zoom/Meet meetings using kdotool (KDE Wayland).

No API keys, app registrations, or admin approval needed.

Requirements:
    sudo dnf install kdotool    # KDE Plasma on Wayland
"""

import re
import subprocess
import sys


def _get_window_titles() -> list[str]:
    """Get all window titles using kdotool (KDE Wayland)."""
    try:
        result = subprocess.run(
            ["kdotool", "search", "--name", ""],
            capture_output=True, text=True, check=True,
        )
        window_ids = result.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    titles = []
    for wid in window_ids:
        try:
            name_result = subprocess.run(
                ["kdotool", "getwindowname", wid],
                capture_output=True, text=True, check=True,
            )
            title = name_result.stdout.strip()
            if title:
                titles.append(title)
        except subprocess.CalledProcessError:
            continue
    return titles


# Patterns to match meeting window/tab titles and extract the meeting name.
# Each is (compiled regex, group index for the meeting subject).
_MEETING_PATTERNS = [
    # Microsoft Teams meeting: "Meeting Name | Microsoft Teams"
    # But NOT chat windows: "Chat | Person Name | Microsoft Teams"
    (re.compile(r"^(?!Chat\b)(.+?)\s*[|–\-]\s*Microsoft Teams", re.IGNORECASE), 1),
    # Zoom: "Meeting Name - Zoom"
    (re.compile(r"^(.+?)\s*[|–\-]\s*Zoom", re.IGNORECASE), 1),
    # Google Meet: "Meeting Name - Google Meet"
    (re.compile(r"^(.+?)\s*[|–\-]\s*Google Meet", re.IGNORECASE), 1),
]

# Titles to skip — generic app UI, not actual meetings
_SKIP_TITLES = {
    "microsoft teams",
    "teams",
    "zoom",
    "google meet",
    "",
}


def get_current_meeting() -> dict | None:
    """
    Detect the current meeting by scanning window titles.

    Returns: {"subject": "Weekly Standup"} or None.
    """
    try:
        titles = _get_window_titles()
    except Exception as e:
        print(f"  ⚠️  Could not read window titles: {e}", file=sys.stderr)
        return None

    if not titles:
        return None

    for title in titles:
        for pattern, group_idx in _MEETING_PATTERNS:
            match = pattern.match(title)
            if match:
                subject = match.group(group_idx).strip()
                if subject.lower() not in _SKIP_TITLES:
                    return {"subject": subject}

    return None


def sanitize_filename(name: str) -> str:
    """Convert a meeting subject into a safe, slug-style filename component."""
    name = re.sub(r'[/\\:*?"<>|]', ' ', name)
    name = re.sub(r'\s+', '-', name.strip())
    name = re.sub(r'[^a-zA-Z0-9\-]', '', name)
    name = re.sub(r'-+', '-', name)
    name = name.strip('-').lower()
    return name[:80] if name else "meeting"
