from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path: str | Path | None, default: Any = None) -> Any:
    if path in {None, ""}:
        return default
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path in {None, ""}:
        return []
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def dump_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=_json_default)
        handle.write("\n")


def dump_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, default=_json_default))
            handle.write("\n")


def resolve_repo_path(repo_root: str | Path, candidate: str | Path | None) -> Path | None:
    if candidate in {None, ""}:
        return None
    path = Path(candidate)
    if path.is_absolute():
        return path
    return Path(repo_root) / path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")
