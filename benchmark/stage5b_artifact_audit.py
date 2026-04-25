from __future__ import annotations

from pathlib import Path
from typing import Any

from .frozen_corpus_support import load_frozen_corpus_mapping
from .io import dump_json, load_json, load_yaml, resolve_repo_path
from .stage5b_metric_pack import STAGE5B_CASE_REQUIREMENTS


ARTIFACT_FILE_MAP = {
    "execution_timeline": "execution_timeline.jsonl",
    "lane_change_events": "lane_change_events.jsonl",
    "lane_change_events_enriched": "lane_change_events_enriched.jsonl",
    "stop_events": "stop_events.jsonl",
    "execution_summary": "execution_summary.json",
    "trajectory_trace": "trajectory_trace.jsonl",
    "control_trace": "control_trace.jsonl",
    "lane_change_phase_trace": "lane_change_phase_trace.jsonl",
    "stop_target": "stop_target.json",
    "planner_quality_summary": "planner_quality_summary.json",
}


def _candidate_stage3c_dirs(repo_root: Path, benchmark_config: dict[str, Any]) -> list[Path]:
    mapping = load_frozen_corpus_mapping(repo_root, benchmark_config)
    candidates: list[Path] = []
    selected_cases = set(
        STAGE5B_CASE_REQUIREMENTS["stage5b_core_golden"]["cases"]
        + STAGE5B_CASE_REQUIREMENTS["stage5b_failure_replay"]["cases"]
    )
    for case_id, entry in (mapping.get("cases") or {}).items():
        if case_id not in selected_cases:
            continue
        replay_mapping = ((entry.get("mapping_modes") or {}).get("replay_regression")) or {}
        stage3c_dir = (((replay_mapping.get("source") or {}).get("stage3c_dir")))
        resolved = resolve_repo_path(repo_root, stage3c_dir)
        if resolved is not None and resolved.exists():
            candidates.append(resolved)
    unique: list[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _artifact_status(stage3c_dirs: list[Path], artifact_key: str) -> tuple[str, list[dict[str, Any]]]:
    filename = ARTIFACT_FILE_MAP[artifact_key]
    evidence: list[dict[str, Any]] = []
    present_count = 0
    approximate = False
    for stage3c_dir in stage3c_dirs:
        artifact_path = stage3c_dir / filename
        info: dict[str, Any] = {
            "stage3c_dir": str(stage3c_dir),
            "artifact_path": str(artifact_path),
            "exists": artifact_path.exists(),
        }
        if artifact_path.exists():
            present_count += 1
            planner_quality_summary = load_json(stage3c_dir / "planner_quality_summary.json", default={}) or {}
            measurement_support = dict(planner_quality_summary.get("measurement_support") or {})
            if artifact_key == "stop_target":
                payload = load_json(artifact_path, default={}) or {}
                support = ((payload.get("measurement_support") or {}).get("stop_overshoot_distance_m")) or "missing"
                info["measurement_support"] = support
                if support == "approximate":
                    approximate = True
            elif artifact_key == "trajectory_trace":
                support = measurement_support.get("trajectory_trace")
                if support:
                    info["measurement_support"] = support
                    approximate = approximate or support == "approximate"
            elif artifact_key == "control_trace":
                support = measurement_support.get("control_trace")
                if support:
                    info["measurement_support"] = support
                    approximate = approximate or support == "approximate"
            elif artifact_key == "lane_change_events_enriched":
                support = measurement_support.get("lane_change_events_enriched")
                if support:
                    info["measurement_support"] = support
            elif artifact_key == "stop_events":
                support = measurement_support.get("stop_events")
                if support:
                    info["measurement_support"] = support
        evidence.append(info)

    if present_count == 0:
        return "missing", evidence
    if artifact_key == "stop_target":
        return ("approximate" if approximate else "present"), evidence
    if present_count < len(stage3c_dirs):
        return "approximate", evidence
    return "present", evidence


def _metric_readiness(artifact_support: dict[str, str]) -> dict[str, Any]:
    def present(name: str) -> bool:
        return artifact_support.get(name) == "present"

    def supported(name: str) -> bool:
        return artifact_support.get(name) in {"present", "approximate"}

    readiness = {
        "lane_change_completion_time_s": "soft_guardrail" if present("lane_change_events_enriched") else "diagnostic_only",
        "lane_change_completion_rate": "hard_gate_ready" if present("lane_change_events_enriched") else "diagnostic_only",
        "lane_change_abort_rate": "soft_guardrail" if present("lane_change_events_enriched") else "diagnostic_only",
        "lane_change_completion_validity": (
            "hard_gate_ready"
            if present("lane_change_phase_trace") and present("planner_quality_summary")
            else "diagnostic_only"
        ),
        "stop_final_error_m": "diagnostic_only",
        "stop_overshoot_distance_m": "soft_guardrail" if supported("stop_target") else "diagnostic_only",
        "stop_hold_stability": "hard_gate_ready" if present("stop_events") else "diagnostic_only",
        "heading_settle_time_s": (
            "soft_guardrail"
            if present("trajectory_trace") and present("lane_change_phase_trace")
            else "diagnostic_only"
        ),
        "lateral_oscillation_rate": "soft_guardrail" if present("control_trace") else "diagnostic_only",
        "trajectory_smoothness_proxy": "soft_guardrail" if present("control_trace") else "diagnostic_only",
        "control_jerk_proxy": "diagnostic_only",
        "planner_execution_latency_ms": "diagnostic_only",
    }
    return readiness


def run_stage5b_artifact_audit(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_config = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    stage3c_dirs = _candidate_stage3c_dirs(repo_root, benchmark_config)
    artifact_support: dict[str, str] = {}
    artifact_evidence: dict[str, Any] = {}
    for artifact_key in ARTIFACT_FILE_MAP:
        status, evidence = _artifact_status(stage3c_dirs, artifact_key)
        artifact_support[artifact_key] = status
        artifact_evidence[artifact_key] = evidence

    payload = {
        "schema_version": "stage5b_artifact_audit_v1",
        "artifact_support": artifact_support,
        "metric_readiness": _metric_readiness(artifact_support),
        "evidence": artifact_evidence,
        "candidate_stage3c_dirs": [str(path) for path in stage3c_dirs],
        "notes": [
            "stop_final_error_m remains diagnostic because replay still lacks an explicit stop-target distance contract",
            "control_trace and trajectory_trace become hard evidence only after Stage 3C current-run replay emits pose/control fields",
            "frozen corpus reference Stage 3C bundles are included in the audit candidate set after materialization",
        ],
    }
    output_path = repo_root / "benchmark" / "stage5b_artifact_audit.json"
    dump_json(output_path, payload)
    return payload
