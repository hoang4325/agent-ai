from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json, load_json


PHASES: tuple[dict[str, Any], ...] = (
    {
        "phase_id": "phase0_nominal_canary",
        "allowed_cases": ["ml_left_positive_core"],
        "objective": "First explicit bounded takeover canary on the cleanest nominal case with no observed fallback during the handoff window.",
        "entry_requirements": [
            "takeover_enable_gate_status=pass",
            "operator_signoff_recorded",
            "kill_switch_verified_before_run",
            "baseline_rollback_path_verified_before_run",
        ],
    },
    {
        "phase_id": "phase1_stop_canary",
        "allowed_cases": ["stop_follow_ambiguity_core"],
        "objective": "Extend explicit bounded takeover canary to one stop-focused case with exact stop target and clean alignment support.",
        "entry_requirements": [
            "phase0_nominal_canary_pass",
            "stop_alignment_ready=true",
            "stop_target_exact=true",
            "stop_final_error_supported=true",
        ],
    },
    {
        "phase_id": "phase2_nominal_ring_extension",
        "allowed_cases": [
            "ml_right_positive_core",
            "junction_straight_core",
            "fr_lc_commit_no_permission",
        ],
        "objective": "Expand bounded authority transfer to the remaining nominal, junction, and failure-replay cases while keeping the same bounded window.",
        "entry_requirements": [
            "phase0_nominal_canary_pass",
            "phase1_stop_canary_pass",
            "authority_window_metrics_within_bound",
        ],
    },
    {
        "phase_id": "phase3_full_ring_candidate",
        "allowed_cases": ["arbitration_stop_during_prepare_right"],
        "objective": "Add the highest-complexity stop/arbitration case last, only after the narrower canaries remain clean.",
        "entry_requirements": [
            "phase2_nominal_ring_extension_pass",
            "organic_stop_fallback_evidence_present",
            "stop_alignment_ready=true",
            "stop_target_exact=true",
        ],
    },
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _make_checklist_item(item_id: str, status: str, summary: str, evidence: Any) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "status": status,
        "summary": summary,
        "evidence": evidence,
    }


def run_stage6_takeover_rollout_design(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"

    enable_gate_result = load_json(benchmark_root / "stage6_takeover_enable_gate_result.json", default={}) or {}
    enable_gate_report = load_json(benchmark_root / "stage6_takeover_enable_gate_report.json", default={}) or {}
    ring_stability_result = load_json(benchmark_root / "stage6_takeover_ring_6case_stability_result.json", default={}) or {}
    organic_fallback_report = load_json(benchmark_root / "stage6_organic_fallback_ring_report.json", default={}) or {}
    timeout_fallback_report = load_json(benchmark_root / "stage6_fallback_authority_ring_report.json", default={}) or {}
    non_timeout_fallback_report = load_json(benchmark_root / "stage6_non_timeout_fallback_ring_report.json", default={}) or {}
    phase0_canary_result = load_json(benchmark_root / "stage6_takeover_canary_phase0_result.json", default={}) or {}
    phase0_canary_report = load_json(benchmark_root / "stage6_takeover_canary_phase0_report.json", default={}) or {}
    phase1_canary_result = load_json(benchmark_root / "stage6_takeover_canary_phase1_result.json", default={}) or {}
    phase1_canary_report = load_json(benchmark_root / "stage6_takeover_canary_phase1_report.json", default={}) or {}
    phase2_result = load_json(benchmark_root / "stage6_takeover_phase2_result.json", default={}) or {}
    phase2_report = load_json(benchmark_root / "stage6_takeover_phase2_report.json", default={}) or {}
    phase3_result = load_json(benchmark_root / "stage6_takeover_phase3_result.json", default={}) or {}
    phase3_report = load_json(benchmark_root / "stage6_takeover_phase3_report.json", default={}) or {}

    derived_metrics = dict(enable_gate_report.get("derived_metrics") or {})
    handoff_duration_p95 = derived_metrics.get("handoff_duration_p95_ticks")
    rollback_delay_p95 = derived_metrics.get("rollback_delay_p95_ticks")
    handoff_distance_p95 = derived_metrics.get("handoff_distance_p95_m")
    stop_case_support = dict(derived_metrics.get("stop_case_support") or {})

    checklist = [
        _make_checklist_item(
            "takeover_enable_gate_pass",
            "done" if str(enable_gate_result.get("takeover_enable_gate_status")) == "pass" else "blocked",
            "Consolidated takeover-enable gate is clean.",
            {"result_path": str(benchmark_root / "stage6_takeover_enable_gate_result.json")},
        ),
        _make_checklist_item(
            "six_case_ring_repeatable",
            "done" if str(ring_stability_result.get("overall_status")) == "ready" else "blocked",
            "Bounded six-case takeover ring is repeatable.",
            {"result_path": str(benchmark_root / "stage6_takeover_ring_6case_stability_result.json")},
        ),
        _make_checklist_item(
            "bounded_authority_window",
            "done"
            if handoff_duration_p95 == 3.0 and rollback_delay_p95 == 1.0 and float(handoff_distance_p95 or 0.0) <= 0.10
            else "blocked",
            "Bounded authority window remains within the designed envelope.",
            {
                "handoff_duration_p95_ticks": handoff_duration_p95,
                "rollback_delay_p95_ticks": rollback_delay_p95,
                "handoff_distance_p95_m": handoff_distance_p95,
            },
        ),
        _make_checklist_item(
            "stop_case_support_clean",
            "done"
            if all(
                float((payload or {}).get("stop_alignment_ready_rate") or 0.0) == 1.0
                and float((payload or {}).get("stop_alignment_consistent_rate") or 0.0) == 1.0
                and float((payload or {}).get("stop_target_exact_rate") or 0.0) == 1.0
                and float((payload or {}).get("stop_final_error_supported_rate") or 0.0) == 1.0
                for payload in stop_case_support.values()
            )
            else "blocked",
            "Stop-focused takeover cases keep exact stop support.",
            stop_case_support,
        ),
        _make_checklist_item(
            "timeout_fallback_ring_ready",
            "done" if bool((timeout_fallback_report.get("aggregate") or {}).get("ready")) else "blocked",
            "Timeout fallback authority ring is clean.",
            dict(timeout_fallback_report.get("aggregate") or {}),
        ),
        _make_checklist_item(
            "non_timeout_fallback_ring_ready",
            "done" if bool((non_timeout_fallback_report.get("aggregate") or {}).get("ready")) else "blocked",
            "Non-timeout fallback authority ring is clean.",
            dict(non_timeout_fallback_report.get("aggregate") or {}),
        ),
        _make_checklist_item(
            "organic_fallback_candidate_observation_ready",
            "done" if bool((organic_fallback_report.get("aggregate") or {}).get("ready")) else "blocked",
            "Organic fallback candidate observation is present on stop-focused cases.",
            dict(organic_fallback_report.get("aggregate") or {}),
        ),
        _make_checklist_item(
            "operator_signoff_recorded",
            "done"
            if bool(((phase3_report.get("operator_signoff") or {}).get("recorded")))
            or bool(((phase2_report.get("operator_signoff") or {}).get("recorded")))
            or bool(((phase1_canary_report.get("operator_signoff") or {}).get("recorded")))
            or bool(((phase0_canary_report.get("operator_signoff") or {}).get("recorded")))
            else "pending",
            "Operational signoff is still required before any explicit takeover canary is executed.",
            (
                phase3_report.get("operator_signoff")
                or phase2_report.get("operator_signoff")
                or phase1_canary_report.get("operator_signoff")
                or phase0_canary_report.get("operator_signoff")
                or None
            ),
        ),
        _make_checklist_item(
            "kill_switch_verified_before_run",
            "done"
            if bool(((phase3_report.get("kill_switch_verification") or {}).get("recorded")))
            or bool(((phase2_report.get("kill_switch_verification") or {}).get("recorded")))
            or bool(((phase1_canary_report.get("kill_switch_verification") or {}).get("recorded")))
            or bool(((phase0_canary_report.get("kill_switch_verification") or {}).get("recorded")))
            else "pending",
            "Per-run verification of abort switch and immediate rollback path is still required.",
            (
                phase3_report.get("kill_switch_verification")
                or phase2_report.get("kill_switch_verification")
                or phase1_canary_report.get("kill_switch_verification")
                or phase0_canary_report.get("kill_switch_verification")
                or None
            ),
        ),
        _make_checklist_item(
            "phase0_nominal_canary_executed",
            "done" if str(phase0_canary_result.get("overall_status")) == "pass" else "pending",
            "Phase0 nominal canary must pass before stop-focused canary work.",
            {
                "result_path": str(benchmark_root / "stage6_takeover_canary_phase0_result.json"),
                "overall_status": phase0_canary_result.get("overall_status"),
                "next_scope_status": phase0_canary_result.get("next_scope_status"),
            }
            if phase0_canary_result
            else None,
        ),
        _make_checklist_item(
            "phase1_stop_canary_executed",
            "done" if str(phase1_canary_result.get("overall_status")) == "pass" else "pending",
            "Phase1 stop canary must pass before nominal ring extension work.",
            {
                "result_path": str(benchmark_root / "stage6_takeover_canary_phase1_result.json"),
                "overall_status": phase1_canary_result.get("overall_status"),
                "next_scope_status": phase1_canary_result.get("next_scope_status"),
            }
            if phase1_canary_result
            else None,
        ),
        _make_checklist_item(
            "phase2_nominal_ring_extension_executed",
            "done" if str(phase2_result.get("overall_status")) == "pass" else "pending",
            "Phase2 nominal ring extension must pass before the final promoted-ring stop/arbitration case.",
            {
                "result_path": str(benchmark_root / "stage6_takeover_phase2_result.json"),
                "overall_status": phase2_result.get("overall_status"),
                "next_scope_status": phase2_result.get("next_scope_status"),
                "passed_case_count": ((phase2_report.get("aggregate") or {}).get("passed_case_count")),
            }
            if phase2_result
            else None,
        ),
        _make_checklist_item(
            "phase3_full_ring_candidate_executed",
            "done" if str(phase3_result.get("overall_status")) == "pass" else "pending",
            "Phase3 promoted-ring stop/arbitration candidate must pass before any end-to-end completion review.",
            {
                "result_path": str(benchmark_root / "stage6_takeover_phase3_result.json"),
                "overall_status": phase3_result.get("overall_status"),
                "next_scope_status": phase3_result.get("next_scope_status"),
                "passed_case_count": ((phase3_report.get("aggregate") or {}).get("passed_case_count")),
            }
            if phase3_result
            else None,
        ),
    ]

    design = {
        "schema_version": "stage6_takeover_rollout_design_v1",
        "generated_at_utc": _utc_now_iso(),
        "design_dependency": {
            "takeover_enable_gate_result_path": str(benchmark_root / "stage6_takeover_enable_gate_result.json"),
            "takeover_enable_gate_status": enable_gate_result.get("takeover_enable_gate_status"),
            "next_scope_status": enable_gate_result.get("next_scope_status"),
            "takeover_enable_status": enable_gate_result.get("takeover_enable_status"),
        },
        "rollout_policy": {
            "rollout_mode": "explicit_takeover_rollout_design_only",
            "takeover_enable_status": "not_ready",
            "default_control_owner": "baseline_controller",
            "authority_window": {
                "handoff_duration_ticks": 3,
                "max_speed_mps": 0.75,
                "min_speed_mps": 0.05,
                "rollback_delay_ticks_max": 1,
                "post_handoff_baseline_tail_ticks_min": 5,
            },
            "hard_abort_criteria": [
                "over_budget=true during authority window",
                "perception_hard_stale health flag present",
                "stage1_worker_failed health flag present",
                "rollback_to_baseline_observed=false",
                "rollback_delay_ticks > 1",
                "handoff_duration_ticks_observed > 3",
                "handoff_distance_estimate_m > 0.10",
                "fallback_success_during_handoff=false when fallback_activated_during_handoff_count>0",
                "stop_alignment_ready=false on stop-focused rollout case",
                "stop_target_exact=false on stop-focused rollout case",
            ],
            "kill_switches": [
                "disable explicit takeover flag before Stage 4 launch",
                "force immediate baseline rollback on first hard-abort condition",
                "terminate CARLA case and mark rollout attempt invalid on infra instability",
            ],
            "authority_transition": [
                "baseline_controller_only",
                "takeover_arm_for_allowlisted_case",
                "shadow_authority_window_bounded_to_3_ticks",
                "immediate_rollback_to_baseline",
                "baseline_post_handoff_tail_observation",
            ],
        },
        "rollout_phases": list(PHASES),
        "notes": [
            "This is rollout design only. It does not authorize or execute takeover rollout.",
            "The first explicit canary should start on ml_left_positive_core, not on the full promoted ring.",
            "The highest-complexity stop/arbitration case stays in the last rollout phase on purpose.",
        ],
    }

    checklist_payload = {
        "schema_version": "stage6_takeover_rollout_checklist_v1",
        "generated_at_utc": design["generated_at_utc"],
        "items": checklist,
    }

    remaining_blockers = [
        str(item["item_id"])
        for item in checklist
        if str(item["status"]) == "blocked"
    ]
    pending_operational_items = [
        str(item["item_id"])
        for item in checklist
        if str(item["status"]) == "pending"
    ]

    result = {
        "schema_version": "stage6_takeover_rollout_result_v1",
        "generated_at_utc": design["generated_at_utc"],
        "rollout_design_status": "ready" if not remaining_blockers else "not_ready",
        "next_scope_status": (
            "ready_for_mpc_e2e_completion_gate_review"
            if str(phase3_result.get("overall_status")) == "pass" and not remaining_blockers
            else "ready_for_phase3_full_ring_candidate_review"
            if str(phase2_result.get("overall_status")) == "pass" and not remaining_blockers
            else "ready_for_phase2_nominal_ring_extension_review"
            if str(phase1_canary_result.get("overall_status")) == "pass" and not remaining_blockers
            else "ready_for_phase1_stop_canary_review"
            if str(phase0_canary_result.get("overall_status")) == "pass" and not remaining_blockers
            else "ready_for_single_case_takeover_canary_review" if not remaining_blockers else "not_ready"
        ),
        "takeover_enable_status": "not_ready",
        "remaining_blockers": remaining_blockers,
        "pending_operational_items": pending_operational_items,
        "design_path": str(benchmark_root / "stage6_takeover_rollout_design.json"),
        "checklist_path": str(benchmark_root / "stage6_takeover_rollout_checklist.json"),
    }

    dump_json(benchmark_root / "stage6_takeover_rollout_design.json", design)
    dump_json(benchmark_root / "stage6_takeover_rollout_checklist.json", checklist_payload)
    dump_json(benchmark_root / "stage6_takeover_rollout_result.json", result)
    return {
        "design": design,
        "checklist": checklist_payload,
        "result": result,
    }
