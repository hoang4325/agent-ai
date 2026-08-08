from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


class _AttrDict(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def _to_eval_scope(obj: Any) -> Any:
    if isinstance(obj, dict):
        return _AttrDict({key: _to_eval_scope(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [_to_eval_scope(value) for value in obj]
    return obj


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _candidate_config_chain(config_path: Path) -> List[Path]:
    config_path = config_path.resolve()
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Expected a YAML config, got: {config_path}")

    root = config_path.parents[len(config_path.parents) - 1]
    parts: List[Path] = []
    for parent in reversed(config_path.parents):
        if parent == config_path.parent:
            default_path = parent / "default.yaml"
            if default_path.exists() and default_path.resolve() != config_path:
                parts.append(default_path)
        elif root in parent.parents or parent == root:
            default_path = parent / "default.yaml"
            if default_path.exists():
                parts.append(default_path)
    parts.append(config_path)

    unique: List[Path] = []
    seen = set()
    for path in parts:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _recursive_eval(obj: Any, scope: Dict[str, Any] | None = None) -> Any:
    if scope is None:
        scope = _to_eval_scope(copy.deepcopy(obj))

    if isinstance(obj, dict):
        for key in list(obj.keys()):
            obj[key] = _recursive_eval(obj[key], scope)
    elif isinstance(obj, list):
        for idx, value in enumerate(list(obj)):
            obj[idx] = _recursive_eval(value, scope)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        evaluated = eval(obj[2:-1], scope)
        obj = _recursive_eval(evaluated, scope)
    return obj


def load_recursive_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    merged: Dict[str, Any] = {}
    for path in _candidate_config_chain(config_path):
        with path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle) or {}
        if not isinstance(content, dict):
            raise ValueError(f"Config file must deserialize to a dict: {path}")
        merged = _deep_merge(merged, content)
    return _recursive_eval(merged)


def find_pipeline_step(config: Dict[str, Any], pipeline_name: str) -> Dict[str, Any]:
    for pipeline_key in ("test_pipeline", "train_pipeline"):
        pipeline = config.get(pipeline_key, [])
        for step in pipeline:
            if step.get("type") == pipeline_name:
                return step
    raise KeyError(f"Could not find pipeline step '{pipeline_name}'")
