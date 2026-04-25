from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json


STAGE6_SHADOW_HARD_GATE_METRICS = [
    {
        "metric": "artifact_completeness",
        "threshold": {"min": 1.0},
        "comparison_mode": "absolute",
        "rationale": "Current-run scenario replay artifacts must be complete before shadow evaluation means anything.",
    },
    {
        "metric": "shadow_output_contract_completeness",
        "threshold": {"min": 1.0},
        "comparison_mode": "absolute",
        "rationale": "Shadow mode is not open unless proposal, summary, comparison, and disagreement artifacts all exist.",
    },
    {
        "metric": "shadow_feasibility_rate",
        "threshold": {"min": 0.95},
        "comparison_mode": "absolute",
        "rationale": "Shadow rollout must produce feasible proposals almost all the time on the promoted subset.",
    },
]

STAGE6_SHADOW_SOFT_GUARDRAIL_METRICS = [
    {
        "metric": "shadow_solver_timeout_rate",
        "comparison_mode": "absolute",
        "rationale": "Timeout pressure matters for rollout confidence, but this remains soft until a real MPC solver is wired in.",
    },
    {
        "metric": "shadow_planner_execution_latency_ms",
        "comparison_mode": "absolute",
        "rationale": "Track shadow-side latency without claiming real-time readiness yet.",
    },
    {
        "metric": "shadow_baseline_disagreement_rate",
        "comparison_mode": "trend_only",
        "rationale": "Shadow disagreement is informative for rollout control, but disagreement alone is not a failure.",
    },
    {
        "metric": "shadow_fallback_recommendation_rate",
        "comparison_mode": "trend_only",
        "rationale": "Tracks how often the shadow planner recommends deterministic fallback instead of nominal proposals.",
    },
    {
        "metric": "shadow_stop_final_error_m",
        "comparison_mode": "relative_to_baseline",
        "rationale": "Proposal-direct stop metric is soft-usable on stop-focused cases when a runtime stop target is present, but still not hard because shadow never takes control.",
    },
    {
        "metric": "shadow_stop_overshoot_distance_m",
        "comparison_mode": "relative_to_baseline",
        "rationale": "Proposal-direct stop overshoot remains soft only; it is useful for stop-quality trend comparison before takeover exists.",
    },
    {
        "metric": "shadow_lateral_oscillation_rate",
        "comparison_mode": "relative_to_baseline",
        "rationale": "Derived directly from emitted shadow proposal profiles, so it can be used as a soft proposal-quality comparison.",
    },
    {
        "metric": "shadow_trajectory_smoothness_proxy",
        "comparison_mode": "relative_to_baseline",
        "rationale": "Derived directly from the shadow proposal path/profile and useful as a soft proposal-quality guardrail.",
    },
    {
        "metric": "shadow_control_jerk_proxy",
        "comparison_mode": "relative_to_baseline",
        "rationale": "Derived directly from the shadow proposal speed/control profile and useful as a soft proposal-quality guardrail.",
    },
    {
        "metric": "shadow_realized_support_rate",
        "comparison_mode": "absolute",
        "rationale": "Confirms the shadow proposal is comparable against a realized future window on the same current run.",
    },
    {
        "metric": "shadow_realized_behavior_match_rate",
        "comparison_mode": "absolute",
        "rationale": "Tracks whether the shadow recommendation stays behavior-consistent with the realized short-horizon outcome.",
    },
]

STAGE6_SHADOW_DIAGNOSTIC_METRICS = [
    {
        "metric": "shadow_lane_change_completion_time_s",
        "comparison_mode": "relative_to_baseline",
        "rationale": "Estimated from the proposed path only; still too counterfactual to use as a shadow guardrail.",
    },
    {
        "metric": "shadow_heading_settle_time_s",
        "comparison_mode": "relative_to_baseline",
        "rationale": "Estimated from the proposed path only; still too counterfactual to use as a shadow guardrail.",
    },
    {
        "metric": "shadow_realized_speed_mae_mps",
        "comparison_mode": "trend_only",
        "rationale": "Useful realized short-horizon comparison, but not calibrated enough yet to gate the promoted subset.",
    },
    {
        "metric": "shadow_realized_progress_mae_m",
        "comparison_mode": "trend_only",
        "rationale": "Useful realized short-horizon comparison, but not calibrated enough yet to gate the promoted subset.",
    },
    {
        "metric": "shadow_realized_stop_distance_mae_m",
        "comparison_mode": "trend_only",
        "rationale": "Valuable for stop-focused realism checks, but still influenced by blocker motion and baseline control.",
    },
]


def _shadow_artifacts() -> list[str]:
    return [
        "stage3c_mpc_shadow_proposal",
        "stage3c_mpc_shadow_summary",
        "stage3c_shadow_baseline_comparison",
        "stage3c_shadow_disagreement_events",
    ]


STAGE6_SHADOW_CASE_REQUIREMENTS = {
    "stage6_shadow_golden": {
        "description": "Minimal Stage 6 MPC shadow rollout subset.",
        "required_pass": True,
        "cases": [
            "ml_right_positive_core",
            "stop_follow_ambiguity_core",
        ],
        "case_metric_overrides": {
            "ml_right_positive_core": {
                "mandatory_artifacts_add": _shadow_artifacts(),
                "primary_metrics": [
                    "artifact_completeness",
                    "shadow_output_contract_completeness",
                    "shadow_feasibility_rate",
                ],
                "secondary_metrics": [
                    "shadow_solver_timeout_rate",
                    "shadow_planner_execution_latency_ms",
                    "shadow_baseline_disagreement_rate",
                    "shadow_fallback_recommendation_rate",
                    "shadow_realized_support_rate",
                    "shadow_realized_behavior_match_rate",
                    "shadow_lateral_oscillation_rate",
                    "shadow_lane_change_completion_time_s",
                    "shadow_trajectory_smoothness_proxy",
                    "shadow_control_jerk_proxy",
                ],
                "metric_thresholds": {
                    "shadow_output_contract_completeness": {"min": 1.0},
                    "shadow_feasibility_rate": {"min": 1.0},
                    "shadow_solver_timeout_rate": {"max": 0.0},
                    "shadow_planner_execution_latency_ms": {"max": 4.0},
                    "shadow_baseline_disagreement_rate": {"max": 0.25},
                    "shadow_fallback_recommendation_rate": {"max": 0.2},
                    "shadow_realized_support_rate": {"min": 0.9},
                    "shadow_realized_behavior_match_rate": {"min": 0.8},
                },
            },
            "stop_follow_ambiguity_core": {
                "mandatory_artifacts_add": _shadow_artifacts(),
                "primary_metrics": [
                    "artifact_completeness",
                    "shadow_output_contract_completeness",
                    "shadow_feasibility_rate",
                ],
                "secondary_metrics": [
                    "shadow_solver_timeout_rate",
                    "shadow_planner_execution_latency_ms",
                    "shadow_baseline_disagreement_rate",
                    "shadow_fallback_recommendation_rate",
                    "shadow_realized_support_rate",
                    "shadow_realized_behavior_match_rate",
                    "shadow_stop_final_error_m",
                    "shadow_stop_overshoot_distance_m",
                    "shadow_trajectory_smoothness_proxy",
                ],
                "metric_thresholds": {
                    "shadow_output_contract_completeness": {"min": 1.0},
                    "shadow_feasibility_rate": {"min": 1.0},
                    "shadow_solver_timeout_rate": {"max": 0.0},
                    "shadow_planner_execution_latency_ms": {"max": 4.5},
                    "shadow_baseline_disagreement_rate": {"max": 0.8},
                    "shadow_fallback_recommendation_rate": {"max": 0.5},
                    "shadow_realized_support_rate": {"min": 0.9},
                    "shadow_realized_behavior_match_rate": {"min": 0.8},
                },
            },
        },
    },
}


def build_stage6_shadow_metric_pack(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    metric_pack = {
        "schema_version": "stage6_shadow_metric_pack_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage": "6_shadow",
        "description": (
            "Stage 6 MPC shadow mode rollout pack for the promoted two-case subset. "
            "This pack gates shadow contract completeness and proposal feasibility only. "
            "Proposal-direct shadow quality metrics can now be tracked as soft guardrails, while counterfactual path-only estimates stay diagnostic. "
            "The current rollout uses an OSQP-backed minimal MPC shadow backend and remains proposal-only."
        ),
        "threshold_authoritative_source": {
            "primary": "stage6_shadow_runtime_case_spec.metric_thresholds",
            "fallback": "benchmark_v1.default_metric_thresholds",
            "runner_resolution": "benchmark.runner_core._case_metric_thresholds",
        },
        "hard_gate_metrics": STAGE6_SHADOW_HARD_GATE_METRICS,
        "soft_guardrail_metrics": STAGE6_SHADOW_SOFT_GUARDRAIL_METRICS,
        "diagnostic_metrics": STAGE6_SHADOW_DIAGNOSTIC_METRICS,
        "comparison_metrics": [
            "shadow_stop_final_error_m",
            "shadow_stop_overshoot_distance_m",
            "shadow_lane_change_completion_time_s",
            "shadow_heading_settle_time_s",
            "shadow_lateral_oscillation_rate",
            "shadow_trajectory_smoothness_proxy",
            "shadow_control_jerk_proxy",
            "shadow_planner_execution_latency_ms",
        ],
        "subset_cases": ["ml_right_positive_core", "stop_follow_ambiguity_core"],
        "case_requirements": STAGE6_SHADOW_CASE_REQUIREMENTS,
        "notes": [
            "Baseline remains the only execution/control authority.",
            "Shadow proposals are emitted and benchmarked without takeover.",
            "Proposal-direct shadow quality metrics are soft-usable; counterfactual path-only estimates remain diagnostic.",
        ],
    }
    dump_json(repo_root / "benchmark" / "stage6_shadow_metric_pack.json", metric_pack)
    return metric_pack
