from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent_ai.shared.artifact_io import load_json as _load_json
from agent_ai.shared.artifact_io import load_jsonl as _load_jsonl
from agent_ai.shared.artifact_io import write_json as _write_json
from agent_ai.shared.artifact_io import write_jsonl as _write_jsonl


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path: str | Path | None, default: Any = None) -> Any:
    """Load JSON; return ``default`` when path is empty or missing."""
    return _load_json(path, default=default, missing_ok=True)


def load_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    """Load JSONL; return empty list when path is empty or missing."""
    return _load_jsonl(path, missing_ok=True)


def dump_json(path: str | Path, payload: Any) -> None:
    """Write JSON with trailing newline (benchmark artifact convention)."""
    _write_json(path, payload, ensure_ascii=True, trailing_newline=True)


def dump_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Rewrite a JSONL file (benchmark artifact convention)."""
    _write_jsonl(path, records, ensure_ascii=True)


def resolve_repo_path(repo_root: str | Path, candidate: str | Path | None) -> Path | None:
    if candidate in {None, ""}:
        return None
    path = Path(candidate)
    if path.is_absolute():
        return path
    return Path(repo_root) / path
