from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json, load_json, load_jsonl


def run_stage6_shadow_contract_audit(
    repo_root: str | Path,
    *,
    case_stage3c_dirs: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    case_stage3c_dirs = case_stage3c_dirs or {}

    cases: list[dict[str, Any]] = []
    required_missing_fields: list[str] = []
    baseline_comparison_ready = True
    shadow_output_contract = "present"

    for case_id, stage3c_dir_raw in case_stage3c_dirs.items():
        stage3c_dir = Path(stage3c_dir_raw)
        proposal_rows = load_jsonl(stage3c_dir / "mpc_shadow_proposal.jsonl")
        summary = load_json(stage3c_dir / "mpc_shadow_summary.json", default={}) or {}
        comparison = load_json(stage3c_dir / "shadow_baseline_comparison.json", default={}) or {}
        disagreements = load_jsonl(stage3c_dir / "shadow_disagreement_events.jsonl")

        missing_for_case: list[str] = []
        if not proposal_rows:
            missing_for_case.append("mpc_shadow_proposal")
        if not summary:
            missing_for_case.append("mpc_shadow_summary")
        if not comparison:
            missing_for_case.append("shadow_baseline_comparison")
        if disagreements is None:
            missing_for_case.append("shadow_disagreement_events")

        if missing_for_case:
            shadow_output_contract = "partial" if shadow_output_contract != "missing" else shadow_output_contract
            if len(missing_for_case) >= 4:
                shadow_output_contract = "missing"
            required_missing_fields.extend(f"{case_id}:{item}" for item in missing_for_case)
        if not comparison:
            baseline_comparison_ready = False

        cases.append(
            {
                "case_id": case_id,
                "stage3c_dir": str(stage3c_dir),
                "proposal_rows": len(proposal_rows),
                "summary_present": bool(summary),
                "comparison_present": bool(comparison),
                "disagreement_rows": len(disagreements),
                "contract_completeness": summary.get("shadow_output_contract_completeness"),
                "soft_guardrail_candidate_metrics": list(summary.get("soft_guardrail_candidate_metrics") or []),
                "proposal_direct_metrics": list(summary.get("proposal_direct_metrics") or []),
                "notes": [
                    (
                        "shadow_latency_is_direct_osqp_solver_timing"
                        if bool(summary) and str(summary.get("proposal_source")) == "osqp_mpc_shadow_backend_v1"
                        else (
                            "shadow_latency_is_direct_surrogate_timing"
                            if bool(summary) and str(summary.get("proposal_source")) == "kinematic_mpc_shadow_surrogate_v1"
                            else ("shadow_latency_is_proxy" if bool(summary) else "shadow_summary_missing")
                        )
                    ),
                    (
                        "quality_metric_support_classification_present"
                        if bool(summary) and bool(summary.get("shadow_metric_support"))
                        else (
                            "quality_values_are_predicted_osqp_shadow_backend"
                            if bool(summary) and str(summary.get("proposal_source")) == "osqp_mpc_shadow_backend_v1"
                            else (
                                "quality_values_are_predicted_shadow_surrogate"
                                if bool(summary) and str(summary.get("proposal_source")) == "kinematic_mpc_shadow_surrogate_v1"
                                else ("quality_values_are_baseline_anchored_proxy" if bool(summary) else "shadow_summary_missing")
                            )
                        )
                    ),
                ],
            }
        )

    payload: dict[str, Any] = {
        "schema_version": "stage6_shadow_contract_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "shadow_output_contract": shadow_output_contract,
        "baseline_comparison_ready": baseline_comparison_ready,
        "required_missing_fields": sorted(required_missing_fields),
        "cases": cases,
        "notes": [
            "Shadow rollout is proposal-only; baseline remains the execution authority.",
            "Current shadow rollout uses an OSQP-backed minimal MPC shadow solver; contract audit now distinguishes proposal-direct soft-usable metrics from diagnostic-only estimates.",
        ],
    }
    dump_json(repo_root / "benchmark" / "stage6_shadow_contract_audit.json", payload)
    return payload
