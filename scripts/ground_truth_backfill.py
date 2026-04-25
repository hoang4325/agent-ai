from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.io import dump_json, load_json, load_yaml, resolve_repo_path  # noqa: E402
from benchmark.semantic_artifacts import ensure_semantic_artifacts  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_output() -> Path:
    return REPO_ROOT / "benchmark" / "reports" / "ground_truth_backfill_plan.json"


def _role_bucket(role: str) -> str:
    role_l = role.lower()
    if "blocker" in role_l:
        return "blocker_current_lane"
    if "adjacent" in role_l:
        return "adjacent_lane_actor"
    if "cross" in role_l:
        return "cross_traffic"
    if "pedestrian" in role_l:
        return "pedestrian_hazard"
    if "lead" in role_l or "follow" in role_l:
        return "lead_vehicle"
    if "route" in role_l or "junction" in role_l:
        return "route_critical_actor"
    return "route_critical_actor"


def _case_specs(cases_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for case_path in sorted(cases_dir.glob("*.yaml")):
        payload = load_yaml(case_path)
        payload["_case_path"] = str(case_path)
        rows.append(payload)
    return rows


def _gt_ref_path(scenario_id: str) -> Path:
    return REPO_ROOT / "benchmark" / "reference_artifacts" / scenario_id / "ground_truth_actor_roles.json"


def _build_plan(case_spec: dict[str, Any]) -> dict[str, Any]:
    scenario_id = str(case_spec.get("scenario_id"))
    status = str(case_spec.get("status", "planned"))
    roles_to_backfill = sorted({_role_bucket(str(r)) for r in (case_spec.get("critical_actors") or [])})

    gt_ref = load_json(_gt_ref_path(scenario_id))
    gt_available = bool(gt_ref)
    gt_pending = True if not gt_ref else bool(gt_ref.get("pending_ground_truth_binding", True))

    if status == "planned":
        confidence = "unavailable"
        source = "scenario_manifest_only"
    elif gt_available and not gt_pending:
        confidence = "heuristic"
        source = "reference_ground_truth + replay_artifacts"
    elif gt_available and gt_pending:
        confidence = "unavailable"
        source = "reference_ground_truth_pending"
    else:
        confidence = "unavailable"
        source = "case_spec_only"

    can_auto_backfill = bool(status in {"ready", "provisional"} and gt_available and not gt_pending)

    manual_work_needed: list[str] = []
    if status == "planned":
        manual_work_needed.append("pin_clean_artifact_bundle")
    if not gt_available:
        manual_work_needed.append("create_ground_truth_actor_roles")
    if gt_pending:
        manual_work_needed.append("resolve_pending_ground_truth_binding")
    if confidence != "exact":
        manual_work_needed.append("optional_manifest_actor_id_mapping_for_exact_truth")

    return {
        "scenario_id": scenario_id,
        "status": status,
        "roles_to_backfill": roles_to_backfill,
        "source_of_truth": source,
        "confidence_level": confidence,
        "can_auto_backfill": can_auto_backfill,
        "manual_work_needed": sorted(set(manual_work_needed)),
    }


def _copy_if_exists(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    if src.resolve() == dst.resolve():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _apply_backfill(case_spec: dict[str, Any], generated_root: Path) -> dict[str, Any]:
    scenario_id = str(case_spec.get("scenario_id"))
    semantic_paths, metadata = ensure_semantic_artifacts(
        case_spec=case_spec,
        repo_root=REPO_ROOT,
        generated_root=generated_root,
    )

    ref_dir = REPO_ROOT / "benchmark" / "reference_artifacts" / scenario_id
    copied = {}
    copied["ground_truth_actor_roles"] = _copy_if_exists(
        semantic_paths.get("ground_truth_actor_roles"),
        ref_dir / "ground_truth_actor_roles.json",
    )
    copied["critical_actor_binding"] = _copy_if_exists(
        semantic_paths.get("critical_actor_binding"),
        ref_dir / "critical_actor_binding.jsonl",
    )
    copied["behavior_arbitration_events"] = _copy_if_exists(
        semantic_paths.get("behavior_arbitration_events"),
        ref_dir / "behavior_arbitration_events.jsonl",
    )
    copied["route_session_binding"] = _copy_if_exists(
        semantic_paths.get("route_session_binding"),
        ref_dir / "route_session_binding.json",
    )

    return {
        "scenario_id": scenario_id,
        "copied": copied,
        "metadata": {
            name: {
                "available": meta.available,
                "source": meta.source,
                "notes": meta.notes,
            }
            for name, meta in metadata.items()
        },
    }


def run_backfill(config_path: Path, output_path: Path, apply: bool) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    cases_dir = resolve_repo_path(REPO_ROOT, cfg["cases_dir"])
    if cases_dir is None:
        raise ValueError("cases_dir missing")

    plans = [_build_plan(spec) for spec in _case_specs(cases_dir)]
    applied: list[dict[str, Any]] = []

    if apply:
        generated_root = REPO_ROOT / "benchmark" / "reports" / "_generated_semantics_backfill"
        generated_root.mkdir(parents=True, exist_ok=True)
        by_scenario = {str(spec.get("scenario_id")): spec for spec in _case_specs(cases_dir)}
        for plan in plans:
            if not plan["can_auto_backfill"]:
                continue
            scenario_id = plan["scenario_id"]
            applied.append(_apply_backfill(by_scenario[scenario_id], generated_root))

    payload = {
        "benchmark_version": cfg.get("version"),
        "generated_at": _utc_now(),
        "applied": apply,
        "plans": plans,
        "applied_results": applied,
    }
    dump_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/apply ground-truth and semantic binding backfill plan.")
    parser.add_argument("--config", default=str(REPO_ROOT / "benchmark" / "benchmark_v1.yaml"))
    parser.add_argument("--output", default=str(_default_output()))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    payload = run_backfill(Path(args.config), Path(args.output), args.apply)
    print(json.dumps({
        "applied": payload["applied"],
        "auto_backfill_candidates": sum(1 for p in payload["plans"] if p["can_auto_backfill"]),
        "output": str(Path(args.output)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
