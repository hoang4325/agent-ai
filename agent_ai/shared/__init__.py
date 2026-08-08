"""
Shared, stage-agnostic utilities for Agent-AI.

Canonical location: ``agent_ai.shared`` (legacy import path: ``common``).

Public surface:
  - agent_ai.shared.artifact_io   — JSON / JSONL helpers
  - agent_ai.shared.numeric       — clamp helpers
  - agent_ai.shared.ports         — Protocol interfaces for pluggable pieces
  - agent_ai.shared.env_flags     — shared environment toggles
  - agent_ai.shared.logging_setup — consistent logging.basicConfig wrapper
"""
from __future__ import annotations

from .artifact_io import (
    append_jsonl,
    load_json,
    load_json_object,
    load_jsonl,
    write_json,
    write_jsonl,
)
from .env_flags import mp4_recording_disabled
from .logging_setup import configure_logging
from .numeric import clamp, clamp01

__all__ = [
    "append_jsonl",
    "clamp",
    "clamp01",
    "configure_logging",
    "load_json",
    "load_json_object",
    "load_jsonl",
    "mp4_recording_disabled",
    "write_json",
    "write_jsonl",
]
