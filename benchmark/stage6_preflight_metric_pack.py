"""
Stage 6 Preflight Metric Pack
=============================
Builds a minimal contract-readiness pack for:
- stop_target
- planner/control latency
- fallback events

This is intentionally a preflight pack, not a full Stage 6 gate.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json


HARD_GATE_METRICS = [
    {
        "metric": "stop_target_defined_rate",
        "threshold": {"min": 1.0},
        "rationale": "Stop-focused runs should not emit opaque stop events without any target contract entry.",
    },
    {
        "metric": "fallback_reason_coverage",
        "threshold": {"min": 1.0},
        "rationale": "Every activated fallback should declare why it activated.",
    },
]

SOFT_GUARDRAIL_METRICS = [
    {
        "metric": "planner_execution_latency_ms",
        "rationale": "Useful preflight latency signal, but budget governance is not final yet.",
    },
    {
        "metric": "decision_to_execution_latency_ms",
        "rationale": "Important for MPC shadow readiness; replay remains approximate when online breakdown is absent.",
    },
    {
        "metric": "over_budget_rate",
        "rationale": "Highlights timing pressure without hard-failing until budgets are stabilized.",
    },
    {
        "metric": "fallback_activation_rate",
        "rationale": "Tracks how often the stack leans on deterministic fallback, but higher is not always wrong.",
    },
    {
        "metric": "fallback_missing_when_required_count",
        "rationale": "Important preflight warning, but required-vs-missing still uses heuristics outside online arbitration.",
    },
    {
        "metric": "stop_overshoot_distance_m",
        "rationale": "Still useful, but remains approximate while stop targets are obstacle-distance proxies.",
    },
]

DIAGNOSTIC_METRICS = [
    {
        "metric": "stop_final_error_m",
        "rationale": "Still blocked on stronger explicit stop-target semantics for every stop source.",
    },
    {
        "metric": "fallback_takeover_latency_ms",
        "rationale": "Sparse outside direct online runs; keep diagnostic until more online evidence exists.",
    },
]


CASE_REQUIREMENTS = {
    "replay_reference_set": {
        "set_name": "stage5b_core_golden",
        "description": "Replay reference subset used to surface preflight contract fields in benchmark reports.",
        "mandatory_artifacts_add": [
            "stage3c_stop_target",
            "stage3c_planner_control_latency",
            "stage3c_fallback_events",
            "stage3c_planner_quality_summary",
        ],
        "secondary_metrics_add": [
            "stop_target_defined_rate",
            "stop_final_error_m",
            "stop_overshoot_distance_m",
            "planner_execution_latency_ms",
            "decision_to_execution_latency_ms",
            "over_budget_rate",
            "fallback_activation_rate",
            "fallback_reason_coverage",
            "fallback_takeover_latency_ms",
            "fallback_missing_when_required_count",
        ],
    },
    "online_reference_set": {
        "set_name": "online_e2e_smoke",
        "description": "Online-only evidence source for direct latency/fallback stage breakdowns.",
        "mandatory_artifacts_add": [
            "stage4_online_tick_record",
            "stage4_online_run_summary",
            "stage4_planner_control_latency",
            "stage4_fallback_events",
        ],
        "secondary_metrics_add": [
            "planner_execution_latency_ms",
            "decision_to_execution_latency_ms",
            "over_budget_rate",
            "fallback_activation_rate",
            "fallback_reason_coverage",
            "fallback_takeover_latency_ms",
            "fallback_missing_when_required_count",
        ],
    },
}


def build_stage6_preflight_metric_pack(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    payload = {
        "schema_version": "stage6_preflight_metric_pack_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage": "6_preflight",
        "description": (
            "Minimal MPC-preflight metric pack for stop-target, latency, and fallback contracts. "
            "This is not a Stage 6 gate; it is contract readiness scaffolding."
        ),
        "hard_gate_metrics": HARD_GATE_METRICS,
        "soft_guardrail_metrics": SOFT_GUARDRAIL_METRICS,
        "diagnostic_metrics": DIAGNOSTIC_METRICS,
        "case_requirements": CASE_REQUIREMENTS,
        "notes": [
            "stop_target_defined_rate is hard-ready as a contract-completeness metric, not as a stop-quality metric.",
            "stop_final_error_m stays diagnostic until explicit stop targets are cleaner across obstacle/route/stop-line sources.",
            "latency metrics are usable now, but budget thresholds should stay soft until more online evidence exists.",
            "fallback metrics describe contract hygiene and takeover observability, not planner quality by themselves.",
        ],
    }
    dump_json(repo_root / "benchmark" / "stage6_preflight_metric_pack.json", payload)
    return payload
