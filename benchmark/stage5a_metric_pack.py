"""
Stage 5A Metric Pack Builder
==============================
Produces stage5a_metric_pack.json with:
- hard_gate_metrics (must pass for Stage 5A)
- soft_guardrail_metrics (trend / warning only)
- diagnostic_metrics (observability)
- case_requirements per gate set
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json, load_yaml, resolve_repo_path


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5A Metric Taxonomy
# ══════════════════════════════════════════════════════════════════════════════

STAGE5A_HARD_GATE_METRICS = [
    {
        "metric": "artifact_completeness",
        "threshold": {"min": 1.0},
        "rationale": "All mandatory artifacts must exist. No compromise.",
    },
    {
        "metric": "route_option_match",
        "threshold": {"min": 1.0},
        "rationale": "Route option must match case expectation exactly.",
    },
    {
        "metric": "expected_behavior_presence",
        "threshold": {"min": 1},
        "rationale": "At least one expected behavior must appear in the session.",
    },
    {
        "metric": "forbidden_behavior_count",
        "threshold": {"max": 0},
        "rationale": "Zero tolerance for forbidden behaviors.",
    },
    {
        "metric": "unsafe_lane_change_intent_count",
        "threshold": {"max": 0},
        "rationale": "No lane change against safety gate.",
    },
    {
        "metric": "stop_success_rate",
        "threshold": {"min": 0.8},
        "rationale": "Stop controller must succeed at high rate for ready cases.",
    },
    {
        "metric": "stop_overshoot_rate",
        "threshold": {"max": 0.0},
        "rationale": "Zero overshoot tolerance for Stage 5A.",
    },
    {
        "metric": "lane_change_success_rate",
        "threshold": {"min": 0.5},
        "rationale": "Applicable to LC-positive cases only.",
        "conditional": "case has commit_lane_change in expected_behaviors",
    },
    {
        "metric": "behavior_execution_mismatch_rate",
        "threshold": {"max": 0.35},
        "rationale": "Mismatch between requested and executing behavior must stay below 35%.",
    },
    {
        "metric": "critical_actor_detected_rate",
        "threshold": {"min": 0.7},
        "rationale": "Critical actors must be detected in at least 70% of eligible frames.",
        "conditional": "pending_ground_truth_binding == false",
    },
    {
        "metric": "route_context_consistency",
        "threshold": {"min": 1.0},
        "rationale": "Route binding must be fully consistent with case expectations.",
    },
]

STAGE5A_SOFT_GUARDRAIL_METRICS = [
    {
        "metric": "blocker_binding_accuracy",
        "rationale": "Binding accuracy is informative but still heuristic-based until manifest-stable actor IDs.",
    },
    {
        "metric": "lane_assignment_consistency",
        "rationale": "Lane assignment depends on derived lane projections. Useful trend signal.",
    },
    {
        "metric": "critical_actor_track_continuity_rate",
        "rationale": "Track continuity is a stability signal but sensitive to sparse transitions.",
    },
    {
        "metric": "route_conditioned_override_rate",
        "rationale": "Override rate alone does not classify correctness.",
    },
    {
        "metric": "route_influence_correctness",
        "rationale": "Depends on case-specific expected behavior semantics.",
    },
    {
        "metric": "arbitration_conflict_count",
        "rationale": "Only meaningful when case defines expected/forbidden arbitration actions.",
    },
    {
        "metric": "abort_rate",
        "rationale": "General robustness metric, not directly safety-critical.",
    },
    {
        "metric": "front_hazard_signal_present",
        "rationale": "Sample-based, not sequence-wide. Useful for world model validation.",
    },
    {
        "metric": "mean_lane_change_time_s",
        "rationale": "Trend monitoring only.",
    },
]

STAGE5A_DIAGNOSTIC_METRICS = [
    {
        "metric": "stale_tick_rate",
        "rationale": "Online-only. Not applicable to replay_regression or scenario_replay.",
    },
    {
        "metric": "mean_total_loop_latency_ms",
        "rationale": "Online-only.",
    },
    {
        "metric": "state_reset_count",
        "rationale": "Online-only.",
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# Case requirement definitions for Stage 5A gates
# ══════════════════════════════════════════════════════════════════════════════

STAGE5A_CASE_REQUIREMENTS = {
    "stage5a_core_golden": {
        "description": "Core golden cases that must fully pass for Stage 5A readiness.",
        "required_pass": True,
        "cases": [
            "ml_right_positive_core",
            "ml_left_positive_core",
            "stop_follow_ambiguity_core",
            "junction_straight_core",
            "arbitration_stop_during_prepare_right",
        ],
    },
    "stage5a_failure_replay": {
        "description": "Failure replay cases that catch known Stage 5A regressions.",
        "required_pass": True,
        "cases": [
            "fr_right_commit_superseded_by_stop",
            "fr_blocker_binding_wrong_lane",
            "fr_route_session_mismatch",
            "fr_arbitration_missing_cancel",
            "fr_stop_overshoot_false_pass",
            "fr_lc_commit_no_permission",
        ],
    },
    "stage5a_soft_coverage": {
        "description": "Provisional / planned cases that provide soft coverage signal.",
        "required_pass": False,
        "cases": [
            "adjacent_right_unsafe_hold",
            "adjacent_left_unsafe_hold",
            "late_emerging_obstacle",
            "route_bias_fork_keep_right",
        ],
    },
}


def build_stage5a_metric_pack(
    repo_root: str | Path,
    config_path: str = "benchmark/benchmark_v1.yaml",
) -> dict[str, Any]:
    """Build and persist the Stage 5A metric pack."""
    repo_root = Path(repo_root)

    metric_pack = {
        "schema_version": "stage5a_metric_pack_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage": "5A",
        "description": (
            "Stage 5A gate pack: contract hardening, obstacle binding, "
            "route/session binding, arbitration cleanup, stop/lane-change semantics. "
            "Thresholds in this pack are taxonomy/advisory defaults only; runner enforcement "
            "must resolve thresholds from case_spec.metric_thresholds first."
        ),
        "threshold_authoritative_source": {
            "primary": "case_spec.metric_thresholds",
            "fallback": "benchmark_v1.default_metric_thresholds",
            "runner_resolution": "benchmark.runner_core._case_metric_thresholds",
            "pack_role": "taxonomy_and_default_threshold_hint_only",
        },
        "hard_gate_metrics": STAGE5A_HARD_GATE_METRICS,
        "soft_guardrail_metrics": STAGE5A_SOFT_GUARDRAIL_METRICS,
        "diagnostic_metrics": STAGE5A_DIAGNOSTIC_METRICS,
        "case_requirements": STAGE5A_CASE_REQUIREMENTS,
        "promotion_criteria": {
            "stage5a_ready": [
                "ALL hard_gate_metrics must pass on stage5a_core_golden cases",
                "ALL stage5a_failure_replay cases must pass (no regression)",
                "ZERO unexpected arbitration events on golden cases",
                "blocker_binding_accuracy >= 0.6 (soft, tracked)",
                "route_context_consistency == 1.0 on all route-conditioned cases",
                "stop_overshoot_rate == 0.0 on all stop-focused cases",
                "lane_change_success_rate >= 0.5 on LC-positive cases",
            ],
            "notes": [
                "case_spec.metric_thresholds is authoritative for runner gating",
                "benchmark_v1.default_metric_thresholds is the fallback when a case omits a metric threshold",
                "stage5a_metric_pack thresholds are advisory expectations, not the runner truth source",
                "online_e2e metrics are diagnostic only for Stage 5A",
                "planned cases are skipped and do not affect gate outcome",
                "soft guardrails produce warnings but do not block Stage 5A pass",
            ],
        },
    }

    output_path = repo_root / "benchmark" / "stage5a_metric_pack.json"
    dump_json(output_path, metric_pack)
    print(f"[Stage5A] Metric pack written to {output_path}")
    return metric_pack
