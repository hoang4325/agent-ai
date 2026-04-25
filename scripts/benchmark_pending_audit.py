from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.io import dump_json, load_json, load_yaml, resolve_repo_path  # noqa: E402
from benchmark.metrics import evaluate_case_metrics  # noqa: E402


def _default_output() -> Path:
    return REPO_ROOT / "benchmark" / "reports" / "benchmark_pending_audit.json"


def _load_case_specs(cases_dir: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for case_path in sorted(cases_dir.glob("*.yaml")):
        payload = load_yaml(case_path)
        payload["_case_path"] = str(case_path)
        specs.append(payload)
    return specs


def _reference_ground_truth(repo_root: Path, scenario_id: str) -> dict[str, Any] | None:
    ref = repo_root / "benchmark" / "reference_artifacts" / scenario_id / "ground_truth_actor_roles.json"
    return load_json(ref)


def _recommended_action(case_spec: dict[str, Any], missing_artifacts: list[str], degraded_metrics: list[str], gt_ref: dict[str, Any] | None) -> str:
    status = str(case_spec.get("status", "planned"))
    pending = bool(case_spec.get("pending_ground_truth_binding", True))
    if status == "planned":
        return "keep_planned_until_clean_artifact_bundle_exists"
    if pending and gt_ref and gt_ref.get("pending_ground_truth_binding") is False:
        return "set_pending_ground_truth_binding_false_and_backfill_semantic_artifacts"
    if pending and not gt_ref:
        return "create_ground_truth_actor_roles_or_keep_non_gate"
    if missing_artifacts:
        return "repair_missing_artifacts_before_gate_promotion"
    if degraded_metrics:
        return "keep_case_guardrail_until_semantic_degradation_cleared"
    if status == "provisional":
        return "candidate_for_ready_after_semantic_review"
    return "ready"


def run_audit(config_path: Path, output_path: Path) -> dict[str, Any]:
    benchmark_cfg = load_yaml(config_path)
    metric_registry = load_yaml(resolve_repo_path(REPO_ROOT, benchmark_cfg["metric_registry"]))
    cases_dir = resolve_repo_path(REPO_ROOT, benchmark_cfg["cases_dir"])
    if cases_dir is None:
        raise ValueError("cases_dir is missing")

    generated_root = REPO_ROOT / "benchmark" / "reports" / "_generated_semantics_audit"
    generated_root.mkdir(parents=True, exist_ok=True)

    case_specs = _load_case_specs(cases_dir)
    case_rows: list[dict[str, Any]] = []

    for case_spec in case_specs:
        scenario_id = str(case_spec.get("scenario_id"))
        status = str(case_spec.get("status", "planned"))
        pending = bool(case_spec.get("pending_ground_truth_binding", True))

        metrics, availability, _resolved_paths, _warnings = evaluate_case_metrics(
            case_spec=case_spec,
            metric_registry=metric_registry,
            repo_root=REPO_ROOT,
            generated_root=generated_root,
        )

        selected_metrics = sorted(set((case_spec.get("primary_metrics") or []) + (case_spec.get("secondary_metrics") or [])))
        degraded_metrics = [
            metric_name
            for metric_name in selected_metrics
            if (metrics.get(metric_name) or {}).get("evaluation_state") in {"metric_skipped", "metric_degraded"}
            or (metrics.get(metric_name) or {}).get("value") is None
        ]

        required_metric_artifacts = set()
        for metric_name in selected_metrics:
            required_metric_artifacts.update((metric_registry.get("metrics", {}).get(metric_name, {}) or {}).get("required_artifacts") or [])

        missing_artifacts = sorted(
            {
                *[
                    art
                    for art in (case_spec.get("mandatory_artifacts") or [])
                    if not bool((availability.get(art) or {}).get("available"))
                ],
                *[
                    art
                    for art in required_metric_artifacts
                    if not bool((availability.get(art) or {}).get("available"))
                ],
            }
        )

        gt_ref = _reference_ground_truth(REPO_ROOT, scenario_id)
        gt_ref_pending = None if gt_ref is None else bool(gt_ref.get("pending_ground_truth_binding", True))

        case_rows.append(
            {
                "scenario_id": scenario_id,
                "status": status,
                "pending_ground_truth_binding": pending,
                "ground_truth_reference_pending": gt_ref_pending,
                "missing_artifacts": missing_artifacts,
                "degraded_metrics": degraded_metrics,
                "recommended_action": _recommended_action(case_spec, missing_artifacts, degraded_metrics, gt_ref),
            }
        )

    readiness_counts = {
        "ready": sum(1 for c in case_rows if c["status"] == "ready"),
        "provisional": sum(1 for c in case_rows if c["status"] == "provisional"),
        "planned": sum(1 for c in case_rows if c["status"] == "planned"),
        "pending_ground_truth_binding_true": sum(1 for c in case_rows if c["pending_ground_truth_binding"]),
    }

    payload = {
        "benchmark_version": benchmark_cfg.get("version"),
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "readiness_counts": readiness_counts,
        "cases": case_rows,
    }
    dump_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit pending/provisional benchmark cases.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "benchmark" / "benchmark_v1.yaml"),
        help="Benchmark config path.",
    )
    parser.add_argument(
        "--output",
        default=str(_default_output()),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    payload = run_audit(Path(args.config), Path(args.output))
    print(json.dumps(payload["readiness_counts"], indent=2))
    print(f"audit_output={Path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
