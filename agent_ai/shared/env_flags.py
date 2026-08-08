"""Shared environment-flag helpers."""
from __future__ import annotations

import os


def mp4_recording_disabled() -> bool:
    """
    Temporary safety switch to disable MP4 recording in test runs.

    Set ``AGENTAI_DISABLE_MP4=0`` (or false/no/off) to re-enable recording.
    Default is disabled (``1``).
    """
    return str(os.getenv("AGENTAI_DISABLE_MP4", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
