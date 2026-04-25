from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .io import dump_json, load_json, load_yaml, resolve_repo_path


DEFAULT_STOP_CASES = [
    "stop_follow_ambiguity_core",
    "junction_straight_core",
]


def _latest_stop_target_path(case_id: str, report_roots: Iterable[Path]) -> Path | None:
    for report_root in report_roots:
        candidate = report_root / "case_runs" / case_id / "stage4" / "stop_target.json"
        if candidate.exists():
            return candidate
    return None


def run_stage4_stop_materialization_audit(
    repo_root: str | Path,
    *,
    case_ids: Iterable[str] | None = None,
    report_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_config = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    mapping_path = resolve_repo_path(
        repo_root,
        ((benchmark_config.get("frozen_corpus") or {}).get("mapping_path")) or "benchmark/mapping_benchmark_to_frozen_corpus.json",
    )
    mapping = load_json(mapping_path, default={}) or {}
    selected_case_ids = [str(case_id) for case_id in (case_ids or DEFAULT_STOP_CASES)]

    resolved_report_roots = [Path(path).resolve() for path in (report_roots or [])]
    if not resolved_report_roots:
        latest_roots = sorted(repo_root.glob("benchmark/reports/stage6_preflight_revalidation_*/online_e2e"))
        latest_roots.extend(sorted(repo_root.glob("benchmark/reports/e2e_online_e2e_*")))
        resolved_report_roots = [path.resolve() for path in reversed(latest_roots)]

    case_rows: list[dict[str, Any]] = []
    for case_id in selected_case_ids:
        bundle_dir = resolve_repo_path(
            repo_root,
            (((mapping.get("cases") or {}).get(case_id) or {}).get("bundle_dir")),
        )
        scenario_manifest = load_json((bundle_dir / "scenario_manifest.json") if bundle_dir else None, default={}) or {}
        runner_manifest = load_json((bundle_dir / "scenario_artifacts" / "runner_manifest.json") if bundle_dir else None, default={}) or {}
        stop_target_path = _latest_stop_target_path(case_id, resolved_report_roots)
        stop_target_payload = load_json(stop_target_path, default={}) or {}
        stop_events = list(stop_target_payload.get("stop_events") or [])
        binding_counts: dict[str, int] = {}
        for row in stop_events:
            key = str(row.get("binding_status") or "missing")
            binding_counts[key] = binding_counts.get(key, 0) + 1

        scenario_recipe = dict(scenario_manifest.get("scenario_recipe") or {})
        spawn_recipe = dict(scenario_recipe.get("spawn_recipe") or {})
        corridor = dict(spawn_recipe.get("corridor") or runner_manifest.get("corridor") or {})
        placements = dict(spawn_recipe.get("placements") or runner_manifest.get("placements") or {})
        blocker_actor_spec_available = bool(
            corridor
            and placements
            and (spawn_recipe.get("blocker_actor_id") is not None or runner_manifest.get("blocker_actor_id") is not None)
            and placements.get("blocker_distance_m") is not None
        )
        scenario_type = str(scenario_recipe.get("scenario_type") or runner_manifest.get("scenario_type") or "")
        can_exact_bind = bool(
            blocker_actor_spec_available
            and scenario_type == "stage3a_non_junction_multilane"
            and corridor.get("lane_id") is not None
        )
        current_binding_status = "missing"
        if binding_counts:
            if binding_counts.get("exact", 0) == len(stop_events):
                current_binding_status = "exact"
            elif binding_counts.get("heuristic", 0) + binding_counts.get("exact", 0) == len(stop_events):
                current_binding_status = "heuristic"
            else:
                current_binding_status = "approximate"
        notes: list[str] = []
        if not runner_manifest:
            notes.append("runner_manifest_missing")
        if not blocker_actor_spec_available:
            notes.append("blocker_actor_spawn_recipe_missing")
        if scenario_type != "stage3a_non_junction_multilane":
            notes.append(f"scenario_type_not_supported_for_exact_blocker_materialization:{scenario_type or 'unknown'}")
        if stop_target_path is None:
            notes.append("online_stop_target_artifact_missing")
        case_rows.append(
            {
                "case_id": case_id,
                "runner_manifest_available": bool(runner_manifest),
                "blocker_actor_spec_available": blocker_actor_spec_available,
                "current_binding_status": current_binding_status,
                "can_exact_bind": can_exact_bind,
                "current_stop_target_path": None if stop_target_path is None else str(stop_target_path),
                "binding_counts": binding_counts,
                "notes": notes,
            }
        )

    payload = {
        "schema_version": "stage4_stop_materialization_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "cases": case_rows,
    }
    dump_json(repo_root / "benchmark" / "stage4_stop_materialization_audit.json", payload)
    return payload
