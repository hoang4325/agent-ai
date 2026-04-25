from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .io import load_json, resolve_repo_path


CURRENT_RUN_SOURCE_TYPES = {
    "scenario_replay_current_run_bundle",
    "current_run_artifact_bundle",
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _mapping_path(repo_root: Path, benchmark_config: dict[str, Any]) -> Path | None:
    frozen_cfg = benchmark_config.get("frozen_corpus") or {}
    candidate = frozen_cfg.get("mapping_path") or benchmark_config.get("frozen_corpus_mapping")
    if not candidate:
        default_path = repo_root / "benchmark" / "mapping_benchmark_to_frozen_corpus.json"
        return default_path if default_path.exists() else None
    resolved = resolve_repo_path(repo_root, candidate)
    if resolved is None or not resolved.exists():
        return None
    return resolved


def load_frozen_corpus_mapping(repo_root: Path, benchmark_config: dict[str, Any]) -> dict[str, Any]:
    mapping_path = _mapping_path(repo_root, benchmark_config)
    return load_json(mapping_path, default={}) if mapping_path else {}


def case_uses_current_run_bundle(case_spec: dict[str, Any]) -> bool:
    return str((case_spec.get("source") or {}).get("type", "")) in CURRENT_RUN_SOURCE_TYPES


def overlay_case_with_frozen_corpus(
    *,
    repo_root: Path,
    benchmark_config: dict[str, Any],
    case_spec: dict[str, Any],
    benchmark_mode: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if case_uses_current_run_bundle(case_spec):
        return case_spec, None

    mapping = load_frozen_corpus_mapping(repo_root, benchmark_config)
    cases = mapping.get("cases") or {}
    case_id = str(case_spec.get("scenario_id", ""))
    entry = cases.get(case_id)
    if not isinstance(entry, dict):
        return case_spec, None

    mode_mapping = ((entry.get("mapping_modes") or {}).get(benchmark_mode)) or {}
    if not mode_mapping:
        return case_spec, {
            "status": "unavailable",
            "reason": "mode_mapping_missing",
            "case_id": case_id,
            "benchmark_mode": benchmark_mode,
            "readiness": entry.get("readiness"),
        }

    overlayed = deepcopy(case_spec)
    if isinstance(mode_mapping.get("source"), dict):
        overlayed["source"] = _deep_merge(dict(overlayed.get("source") or {}), mode_mapping["source"])
    if isinstance(mode_mapping.get("scenario_recipe"), dict):
        overlayed["scenario_recipe"] = _deep_merge(
            dict(overlayed.get("scenario_recipe") or {}),
            mode_mapping["scenario_recipe"],
        )
    if isinstance(mode_mapping.get("artifacts"), dict):
        artifacts = dict(overlayed.get("artifacts") or {})
        artifacts.update(mode_mapping["artifacts"])
        overlayed["artifacts"] = artifacts
    if isinstance(mode_mapping.get("trace_requirements"), dict):
        overlayed["trace_requirements"] = _deep_merge(
            dict(overlayed.get("trace_requirements") or {}),
            mode_mapping["trace_requirements"],
        )

    overlayed["frozen_corpus"] = _deep_merge(
        dict(overlayed.get("frozen_corpus") or {}),
        {
            "enabled": True,
            "bundle_version": mapping.get("bundle_version"),
            "corpus_root": mapping.get("corpus_root"),
            "bundle_dir": entry.get("bundle_dir"),
            "case_manifest": entry.get("case_manifest"),
            "scenario_manifest": entry.get("scenario_manifest"),
            "runner_scenario_manifest": entry.get("runner_scenario_manifest"),
            "benchmark_reference_path": entry.get("benchmark_reference_path"),
            "provenance_path": entry.get("provenance_path"),
            "readiness": entry.get("readiness"),
            "overlay_mode": benchmark_mode,
        },
    )

    return overlayed, {
        "status": "applied",
        "case_id": case_id,
        "benchmark_mode": benchmark_mode,
        "readiness": entry.get("readiness"),
        "bundle_dir": entry.get("bundle_dir"),
    }
