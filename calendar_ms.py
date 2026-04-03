#!/usr/bin/env python3
"""
Microsoft 365 Calendar Integration
====================================
Queries the current user's Outlook calendar via Microsoft Graph API
to find the meeting happening right now (or about to start).

Auth: MSAL device code flow — user logs in once via browser, then
tokens are cached and auto-refreshed.

Setup:
    1. Register an app in Azure Portal > App registrations
       - Supported account types: your org (single tenant)
       - Enable "Allow public client flows" under Authentication
       - Add API permission: Calendars.Read (delegated)
    2. Set env var:  export MS_CALENDAR_CLIENT_ID=<your-app-client-id>
       Or write it to ~/.cache/meeting-transcriber/config.json:
       {"client_id": "<your-app-client-id>"}
    3. First run will prompt: "Go to https://microsoft.com/devicelogin ..."

Dependencies:
    pip install msal requests
"""

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCOPES = ["Calendars.Read"]
CACHE_DIR = Path.home() / ".cache" / "meeting-transcriber"
TOKEN_CACHE_PATH = CACHE_DIR / "ms_token_cache.json"
CONFIG_PATH = CACHE_DIR / "config.json"


def _get_client_id() -> str | None:
    """Resolve the Azure AD app client ID from env or config file."""
    client_id = os.environ.get("MS_CALENDAR_CLIENT_ID")
    if client_id:
        return client_id

    if CONFIG_PATH.is_file():
        try:
            with open(CONFIG_PATH) as f:
                config = json.load(f)
            return config.get("client_id")
        except (json.JSONDecodeError, OSError):
            pass

    return None


def _get_msal_app():
    """Create an MSAL PublicClientApplication with persistent token cache."""
    import msal

    client_id = _get_client_id()
    if not client_id:
        return None

    cache = msal.SerializableTokenCache()
    if TOKEN_CACHE_PATH.is_file():
        cache.deserialize(TOKEN_CACHE_PATH.read_text())

    app = msal.PublicClientApplication(
        client_id,
        authority="https://login.microsoftonline.com/common",
        token_cache=cache,
    )
    return app


def _save_cache(app):
    """Persist the MSAL token cache to disk."""
    cache = app.token_cache
    if cache.has_state_changed:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        TOKEN_CACHE_PATH.write_text(cache.serialize())


def _acquire_token(app) -> str | None:
    """Get an access token, using cached credentials or device code flow."""
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            _save_cache(app)
            return result["access_token"]

    # Fall back to device code flow
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        print(f"  ⚠️  Calendar auth failed: {flow.get('error_description', 'unknown error')}", file=sys.stderr)
        return None

    print(f"\n  📅 Calendar sign-in required (one-time):", file=sys.stderr)
    print(f"     {flow['message']}", file=sys.stderr)

    result = app.acquire_token_by_device_flow(flow)
    _save_cache(app)

    if "access_token" in result:
        return result["access_token"]

    print(f"  ⚠️  Calendar auth failed: {result.get('error_description', 'unknown error')}", file=sys.stderr)
    return None


def _query_calendar_view(token: str) -> dict | None:
    """Query Microsoft Graph for meetings around the current time."""
    import requests

    now = datetime.now(timezone.utc)
    start = (now - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%S.0000000")
    end = (now + timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%S.0000000")

    resp = requests.get(
        "https://graph.microsoft.com/v1.0/me/calendarView",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "startDateTime": start,
            "endDateTime": end,
            "$orderby": "start/dateTime desc",
            "$select": "subject,start,end,isOnlineMeeting",
            "$top": 5,
        },
        timeout=10,
    )

    if resp.status_code != 200:
        print(f"  ⚠️  Calendar API error ({resp.status_code}): {resp.text[:200]}", file=sys.stderr)
        return None

    events = resp.json().get("value", [])
    if not events:
        return None

    # Pick the meeting closest to now
    best = None
    best_dist = float("inf")
    for ev in events:
        try:
            ev_start = datetime.fromisoformat(ev["start"]["dateTime"].rstrip("Z")).replace(tzinfo=timezone.utc)
            dist = abs((now - ev_start).total_seconds())
            if dist < best_dist:
                best_dist = dist
                best = ev
        except (KeyError, ValueError):
            continue

    if best:
        return {"subject": best.get("subject", ""), "start": best["start"]["dateTime"]}
    return None


def get_current_meeting() -> dict | None:
    """
    Public API: return the current calendar meeting, or None.

    Returns: {"subject": "Weekly Standup", "start": "2026-04-02T14:00:00"} or None.
    Gracefully returns None on any failure (missing deps, no config, network error).
    """
    try:
        import msal  # noqa: F401
        import requests  # noqa: F401
    except ImportError:
        return None

    try:
        app = _get_msal_app()
        if not app:
            return None

        token = _acquire_token(app)
        if not token:
            return None

        return _query_calendar_view(token)

    except Exception as e:
        print(f"  ⚠️  Calendar lookup failed: {e}", file=sys.stderr)
        return None


def sanitize_filename(name: str) -> str:
    """Convert a meeting subject into a safe, slug-style filename component."""
    # Remove unsafe characters
    name = re.sub(r'[/\\:*?"<>|]', ' ', name)
    # Collapse whitespace to hyphens
    name = re.sub(r'\s+', '-', name.strip())
    # Remove non-alphanumeric chars except hyphens
    name = re.sub(r'[^a-zA-Z0-9\-]', '', name)
    # Collapse multiple hyphens
    name = re.sub(r'-+', '-', name)
    # Strip and lowercase
    name = name.strip('-').lower()
    # Truncate
    return name[:80] if name else "meeting"
