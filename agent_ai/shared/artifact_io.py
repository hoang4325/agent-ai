"""
Shared JSON / JSONL artifact helpers.

Stages historically inlined identical ``_write_json`` / ``_append_jsonl``
helpers. Use this module so new code and refactors share one implementation
without changing on-disk artifact layout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable


def _json_default_strict(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _json_default_permissive(value: Any) -> Any:
    """Fallback used by long-running bridges that may log mixed types."""
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _resolve_default(default: Callable[[Any], Any] | None, *, permissive: bool) -> Callable[[Any], Any]:
    if default is not None:
        return default
    return _json_default_permissive if permissive else _json_default_strict


def write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = True,
    trailing_newline: bool = False,
    permissive: bool = False,
    default: Callable[[Any], Any] | None = None,
) -> None:
    """Write a JSON file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder_default = _resolve_default(default, permissive=permissive)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=indent,
            ensure_ascii=ensure_ascii,
            default=encoder_default,
        )
        if trailing_newline:
            handle.write("\n")


def append_jsonl(
    path: str | Path,
    payload: Any,
    *,
    ensure_ascii: bool = True,
    permissive: bool = False,
    default: Callable[[Any], Any] | None = None,
) -> None:
    """Append one JSON object as a single line (JSONL)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder_default = _resolve_default(default, permissive=permissive)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=ensure_ascii, default=encoder_default))
        handle.write("\n")


def write_jsonl(
    path: str | Path,
    records: Iterable[Any],
    *,
    ensure_ascii: bool = True,
    permissive: bool = False,
    default: Callable[[Any], Any] | None = None,
) -> None:
    """Rewrite a JSONL file with the given records (truncates existing content)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder_default = _resolve_default(default, permissive=permissive)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=ensure_ascii, default=encoder_default))
            handle.write("\n")


def load_json(
    path: str | Path | None,
    default: Any = None,
    *,
    missing_ok: bool = False,
) -> Any:
    """
    Load a JSON file.

    - If ``path`` is None/empty and ``missing_ok`` is True, return ``default``.
    - If the file does not exist and ``missing_ok`` is True, return ``default``.
    - Otherwise raise FileNotFoundError / JSON errors as usual.
    """
    if path in {None, ""}:
        if missing_ok:
            return default
        raise FileNotFoundError("JSON path is empty or None.")
    path = Path(path)
    if not path.exists():
        if missing_ok:
            return default
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_object(
    path: str | Path,
    *,
    missing_ok: bool = False,
    default: Any = None,
) -> dict[str, Any] | Any:
    """Load JSON and require a top-level object (dict), unless missing_ok."""
    payload = load_json(path, default=default, missing_ok=missing_ok)
    if payload is default and missing_ok:
        return default
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_jsonl(
    path: str | Path | None,
    *,
    missing_ok: bool = True,
    objects_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Load a JSONL file into a list of objects.

    Empty lines are skipped. When the file is missing:
      - ``missing_ok=True``  → return ``[]``
      - ``missing_ok=False`` → raise FileNotFoundError

    When ``objects_only`` is True, non-dict rows are skipped.
    """
    if path in {None, ""}:
        if missing_ok:
            return []
        raise FileNotFoundError("JSONL path is empty or None.")
    path = Path(path)
    if not path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if objects_only and not isinstance(row, dict):
                continue
            records.append(row)
    return records


def touch_jsonl(path: str | Path) -> None:
    """Create/truncate an empty JSONL file (and parents)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
