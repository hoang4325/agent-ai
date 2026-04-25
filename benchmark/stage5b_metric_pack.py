"""
Stage 5B Metric Pack Builder
============================
Produces stage5b_metric_pack.json with:
- hard_gate_metrics
- soft_guardrail_metrics
- diagnostic_metrics
- case_requirements for Stage 5B runtime overlays
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json


STAGE5B_HARD_GATE_METRICS = [
    {
        "metric": "artifact_completeness",
        "threshold": {"min": 1.0},
        "rationale": "Current-run planner-quality artifacts must exist before Stage 5B means anything.",
    },
    {
        "metric": "lane_change_completion_rate",
        "threshold": {"min": 0.5},
        "rationale": "Positive lane-change cases must materially complete, not just request or start the maneuver.",
    },
    {
        "metric": "lane_change_completion_validity",
        "threshold": {"min": 1.0},
        "rationale": "Completed lane changes must land on the intended lane with valid completion evidence.",
    },
    {
        "metric": "stop_hold_stability",
        "threshold": {"min": 0.9},
        "rationale": "Stop-focused cases must hold a stable stop once reached.",
    },
]

STAGE5B_SOFT_GUARDRAIL_METRICS = [
    {
        "metric": "lane_change_completion_time_s",
        "rationale": "Planner-quality trend metric for completion speed; useful but still sensitive to replay timing.",
    },
    {
        "metric": "lane_change_abort_rate",
        "rationale": "Tracks preemption/abort behavior quality without hard-failing every abort.",
    },
    {
        "metric": "stop_overshoot_distance_m",
        "rationale": "Good stop-quality proxy, but still approximate until explicit stop-target binding is cleaner.",
    },
    {
        "metric": "heading_settle_time_s",
        "rationale": "Post-lane-change settle behavior is valuable quality signal, but heuristic-driven.",
    },
    {
        "metric": "lateral_oscillation_rate",
        "rationale": "Control cleanliness proxy from replay control trace.",
    },
    {
        "metric": "trajectory_smoothness_proxy",
        "rationale": "Planner/execution smoothness proxy from control deltas and speed profile.",
    },
]

STAGE5B_DIAGNOSTIC_METRICS = [
    {
        "metric": "stop_final_error_m",
        "rationale": "Blocked on a stronger explicit stop-target contract.",
    },
    {
        "metric": "control_jerk_proxy",
        "rationale": "Helpful control roughness signal, but still proxy-only.",
    },
    {
        "metric": "planner_execution_latency_ms",
        "rationale": "Online-only for now; not part of the primary Stage 5B replay path.",
    },
]


def _planner_quality_artifacts() -> list[str]:
    return [
        "stage3c_execution_summary",
        "stage3c_execution_timeline",
        "stage3c_lane_change_events",
        "stage3c_lane_change_events_enriched",
        "stage3c_stop_events",
        "stage3c_trajectory_trace",
        "stage3c_control_trace",
        "stage3c_lane_change_phase_trace",
        "stage3c_stop_target",
        "stage3c_planner_quality_summary",
    ]


STAGE5B_CASE_REQUIREMENTS = {
    "stage5b_core_golden": {
        "description": "Planner / execution quality golden subset for Stage 5B.",
        "required_pass": True,
        "cases": [
            "ml_right_positive_core",
            "ml_left_positive_core",
            "stop_follow_ambiguity_core",
            "junction_straight_core",
        ],
        "case_metric_overrides": {
            "ml_right_positive_core": {
                "mandatory_artifacts_add": _planner_quality_artifacts(),
                "primary_metrics": [
                    "artifact_completeness",
                    "lane_change_completion_rate",
                    "lane_change_completion_validity",
                ],
                "secondary_metrics": [
                    "lane_change_completion_time_s",
                    "lateral_oscillation_rate",
                    "trajectory_smoothness_proxy",
                    "control_jerk_proxy",
                ],
                "metric_thresholds": {
                    "lane_change_completion_rate": {"min": 0.5},
                    "lane_change_completion_validity": {"min": 1.0},
                    "lane_change_completion_time_s": {"max": 5.5},
                    "lateral_oscillation_rate": {"max": 0.30},
                    "trajectory_smoothness_proxy": {"max": 0.40},
                },
            },
            "ml_left_positive_core": {
                "mandatory_artifacts_add": _planner_quality_artifacts(),
                "primary_metrics": [
                    "artifact_completeness",
                    "lane_change_completion_rate",
                    "lane_change_completion_validity",
                ],
                "secondary_metrics": [
                    "lane_change_completion_time_s",
                    "lateral_oscillation_rate",
                    "trajectory_smoothness_proxy",
                    "control_jerk_proxy",
                ],
                "metric_thresholds": {
                    "lane_change_completion_rate": {"min": 0.5},
                    "lane_change_completion_validity": {"min": 1.0},
                    "lane_change_completion_time_s": {"max": 5.5},
                    "lateral_oscillation_rate": {"max": 0.30},
                    "trajectory_smoothness_proxy": {"max": 0.40},
                },
            },
            "stop_follow_ambiguity_core": {
                "mandatory_artifacts_add": _planner_quality_artifacts(),
                "primary_metrics": [
                    "artifact_completeness",
                    "stop_hold_stability",
                ],
                "secondary_metrics": [
                    "trajectory_smoothness_proxy",
                    "control_jerk_proxy",
                ],
                "metric_thresholds": {
                    "stop_hold_stability": {"min": 0.75},
                    "trajectory_smoothness_proxy": {"max": 0.45},
                },
            },
            "junction_straight_core": {
                "mandatory_artifacts_add": _planner_quality_artifacts(),
                "primary_metrics": [
                    "artifact_completeness",
                ],
                "secondary_metrics": [
                    "trajectory_smoothness_proxy",
                    "lateral_oscillation_rate",
                    "control_jerk_proxy",
                ],
                "metric_thresholds": {
                    "trajectory_smoothness_proxy": {"max": 0.45},
                    "lateral_oscillation_rate": {"max": 0.35},
                },
            },
        },
    },
    "stage5b_failure_replay": {
        "description": "Failure replay subset for Stage 5B planner/execution regressions.",
        "required_pass": True,
        "cases": [
            "fr_right_commit_superseded_by_stop",
            "fr_stop_overshoot_false_pass",
            "fr_lc_commit_no_permission",
        ],
        "case_metric_overrides": {
            "fr_right_commit_superseded_by_stop": {
                "mandatory_artifacts_add": _planner_quality_artifacts(),
                "primary_metrics": [
                    "artifact_completeness",
                    "lane_change_abort_rate",
                    "stop_hold_stability",
                ],
                "secondary_metrics": [
                    "trajectory_smoothness_proxy",
                    "control_jerk_proxy",
                ],
                "metric_thresholds": {
                    "lane_change_abort_rate": {"min": 0.5},
                    "stop_hold_stability": {"min": 0.75},
                    "trajectory_smoothness_proxy": {"max": 0.50},
                },
            },
            "fr_stop_overshoot_false_pass": {
                "mandatory_artifacts_add": _planner_quality_artifacts(),
                "primary_metrics": [
                    "artifact_completeness",
                    "stop_hold_stability",
                ],
                "secondary_metrics": [
                    "trajectory_smoothness_proxy",
                    "control_jerk_proxy",
                ],
                "metric_thresholds": {
                    "stop_hold_stability": {"min": 0.75},
                    "trajectory_smoothness_proxy": {"max": 0.45},
                },
            },
            "fr_lc_commit_no_permission": {
                "mandatory_artifacts_add": _planner_quality_artifacts(),
                "primary_metrics": [
                    "artifact_completeness",
                    "lane_change_completion_rate",
                    "lane_change_completion_validity",
                ],
                "secondary_metrics": [
                    "lane_change_abort_rate",
                    "lane_change_completion_time_s",
                    "trajectory_smoothness_proxy",
                    "control_jerk_proxy",
                ],
                "metric_thresholds": {
                    "lane_change_completion_rate": {"min": 0.5},
                    "lane_change_completion_validity": {"min": 1.0},
                    "lane_change_abort_rate": {"max": 0.5},
                    "lane_change_completion_time_s": {"max": 5.5},
                    "trajectory_smoothness_proxy": {"max": 0.45},
                },
            },
        },
    },
}


def build_stage5b_metric_pack(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    metric_pack = {
        "schema_version": "stage5b_metric_pack_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage": "5B",
        "description": (
            "Stage 5B gate pack for planner/execution quality on top of frozen corpus + scenario_replay. "
            "Thresholds in this pack are advisory defaults for the Stage 5B runtime overlay; "
            "runner enforcement still resolves thresholds from case_spec.metric_thresholds."
        ),
        "threshold_authoritative_source": {
            "primary": "stage5b_runtime_case_spec.metric_thresholds",
            "fallback": "benchmark_v1.default_metric_thresholds",
            "runner_resolution": "benchmark.runner_core._case_metric_thresholds",
            "pack_role": "taxonomy_and_stage5b_runtime_overlay_defaults",
        },
        "hard_gate_metrics": STAGE5B_HARD_GATE_METRICS,
        "soft_guardrail_metrics": STAGE5B_SOFT_GUARDRAIL_METRICS,
        "diagnostic_metrics": STAGE5B_DIAGNOSTIC_METRICS,
        "case_requirements": STAGE5B_CASE_REQUIREMENTS,
        "promotion_criteria": {
            "stage5b_ready": [
                "All hard Stage 5B metrics pass on stage5b_core_golden",
                "All stage5b_failure_replay cases pass with expected abort/hold quality behavior",
                "No hard-gate metric is evaluated from missing artifacts",
                "Stop final error remains diagnostic-only until an explicit stop-target contract is available",
            ],
            "notes": [
                "Stage 5B is planner / execution quality only",
                "Do not promote jerk or latency to hard gate yet",
                "Scenario replay is the primary gate driver; replay_regression remains supporting reference",
            ],
        },
    }
    output_path = repo_root / "benchmark" / "stage5b_metric_pack.json"
    dump_json(output_path, metric_pack)
    print(f"[Stage5B] Metric pack written to {output_path}")
    return metric_pack
