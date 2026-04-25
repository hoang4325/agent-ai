from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage3c_coverage.planner_quality import load_jsonl as load_stage3c_jsonl
from stage3c_coverage.planner_quality import write_planner_quality_artifacts

from .io import dump_json, load_json, load_yaml, resolve_repo_path


def _read_set_cases(repo_root: Path, benchmark_config: dict[str, Any], set_name: str) -> list[str]:
    set_path = resolve_repo_path(repo_root, (benchmark_config.get("sets") or {}).get(set_name))
    payload = load_yaml(set_path) if set_path else {}
    return list(payload.get("cases") or [])


def _candidate_replay_stage3c_dirs(repo_root: Path, benchmark_config: dict[str, Any]) -> list[Path]:
    mapping_path = resolve_repo_path(
        repo_root,
        ((benchmark_config.get("frozen_corpus") or {}).get("mapping_path")) or "benchmark/mapping_benchmark_to_frozen_corpus.json",
    )
    mapping = load_json(mapping_path, default={}) or {}
    selected_cases = set(
        _read_set_cases(repo_root, benchmark_config, "stage5b_core_golden")
        + _read_set_cases(repo_root, benchmark_config, "stage5b_failure_replay")
    )
    dirs: list[Path] = []
    for case_id, entry in (mapping.get("cases") or {}).items():
        if case_id not in selected_cases:
            continue
        replay_mapping = ((entry.get("mapping_modes") or {}).get("replay_regression")) or {}
        stage3c_dir = (((replay_mapping.get("source") or {}).get("stage3c_dir")))
        resolved = resolve_repo_path(repo_root, stage3c_dir)
        if resolved is not None and resolved.exists():
            dirs.append(resolved)
    return sorted({path.resolve() for path in dirs}, key=lambda p: str(p))


def _candidate_online_stage4_dirs(repo_root: Path) -> list[Path]:
    report_roots = sorted(repo_root.glob("benchmark/reports/e2e_online_e2e_*"))
    report_roots.extend(sorted(repo_root.glob("benchmark/reports/stage6_preflight_revalidation_*/online_e2e")))
    dirs: list[Path] = []
    for report_root in report_roots:
        case_runs_dir = report_root / "case_runs"
        if not case_runs_dir.exists():
            continue
        for case_dir in case_runs_dir.iterdir():
            stage4_dir = case_dir / "stage4"
            if stage4_dir.exists():
                dirs.append(stage4_dir)
    return sorted({path.resolve() for path in dirs}, key=lambda p: str(p))


def _candidate_current_replay_stage3c_dirs(repo_root: Path) -> list[Path]:
    report_roots = sorted(repo_root.glob("benchmark/reports/e2e_scenario_replay_*"))
    report_roots.extend(sorted(repo_root.glob("benchmark/reports/stage6_preflight_revalidation_*/scenario_replay")))
    if not report_roots:
        return []
    latest_root = report_roots[-1]
    dirs: list[Path] = []
    case_runs_dir = latest_root / "case_runs"
    if not case_runs_dir.exists():
        return []
    for case_dir in case_runs_dir.iterdir():
        stage3c_dir = case_dir / "stage3c"
        if stage3c_dir.exists():
            dirs.append(stage3c_dir)
    return sorted({path.resolve() for path in dirs}, key=lambda p: str(p))


def _backfill_stage3c_dir(stage3c_dir: Path) -> None:
    execution_path = stage3c_dir / "execution_timeline.jsonl"
    lane_change_path = stage3c_dir / "lane_change_events.jsonl"
    if not execution_path.exists() or not lane_change_path.exists():
        return
    write_planner_quality_artifacts(
        stage3c_dir,
        execution_records=load_stage3c_jsonl(execution_path),
        raw_lane_change_events=load_stage3c_jsonl(lane_change_path),
        source_stage="stage3c_replay",
    )


def _backfill_stage4_dir(stage4_dir: Path) -> None:
    execution_path = stage4_dir / "execution_timeline.jsonl"
    lane_change_path = stage4_dir / "lane_change_events.jsonl"
    tick_path = stage4_dir / "online_tick_record.jsonl"
    if not execution_path.exists() or not lane_change_path.exists():
        return
    write_planner_quality_artifacts(
        stage4_dir,
        execution_records=load_stage3c_jsonl(execution_path),
        raw_lane_change_events=load_stage3c_jsonl(lane_change_path),
        online_tick_records=load_stage3c_jsonl(tick_path) if tick_path.exists() else [],
        source_stage="stage4_online",
    )


def _contract_status(values: list[str]) -> str:
    normalized = [value for value in values if value]
    if not normalized:
        return "missing"
    if any(value == "missing" for value in normalized):
        return "approximate"
    if any(value == "approximate" for value in normalized):
        return "approximate"
    return "present"


def _stop_target_status(
    frozen_stage3c_dirs: list[Path],
    current_stage3c_dirs: list[Path],
    stage4_dirs: list[Path],
) -> tuple[str, dict[str, Any]]:
    frozen_evidence: list[dict[str, Any]] = []
    current_evidence: list[dict[str, Any]] = []
    online_evidence: list[dict[str, Any]] = []
    statuses: list[str] = []
    for stage3c_dir in frozen_stage3c_dirs:
        payload = load_json(stage3c_dir / "stop_target.json", default={}) or {}
        support = dict(payload.get("measurement_support") or {})
        summary = dict(payload.get("summary") or {})
        status = "missing"
        if payload:
            defined_support = str(support.get("stop_target_defined_rate") or "missing")
            final_error_support = str(support.get("stop_final_error_m") or "missing")
            status = "present" if defined_support == "present" and final_error_support == "present" else "approximate"
        statuses.append(status)
        frozen_evidence.append(
            {
                "stage3c_dir": str(stage3c_dir),
                "exists": bool(payload),
                "measurement_support": support,
                "summary": {
                    "stop_target_defined_rate": summary.get("stop_target_defined_rate"),
                    "explicit_target_events": summary.get("explicit_target_events"),
                    "mean_stop_final_error_m": summary.get("mean_stop_final_error_m"),
                },
                "status": status,
            }
        )
    for stage3c_dir in current_stage3c_dirs:
        payload = load_json(stage3c_dir / "stop_target.json", default={}) or {}
        support = dict(payload.get("measurement_support") or {})
        summary = dict(payload.get("summary") or {})
        status = "missing"
        if payload:
            defined_support = str(support.get("stop_target_defined_rate") or "missing")
            final_error_support = str(support.get("stop_final_error_m") or "missing")
            status = "present" if defined_support == "present" and final_error_support == "present" else "approximate"
        statuses.append(status)
        current_evidence.append(
            {
                "stage3c_dir": str(stage3c_dir),
                "exists": bool(payload),
                "measurement_support": support,
                "summary": {
                    "stop_target_defined_rate": summary.get("stop_target_defined_rate"),
                    "explicit_target_events": summary.get("explicit_target_events"),
                    "mean_stop_final_error_m": summary.get("mean_stop_final_error_m"),
                },
                "status": status,
            }
        )
    for stage4_dir in stage4_dirs:
        payload = load_json(stage4_dir / "stop_target.json", default={}) or {}
        support = dict(payload.get("measurement_support") or {})
        summary = dict(payload.get("summary") or {})
        status = "missing"
        if payload:
            defined_support = str(support.get("stop_target_defined_rate") or "missing")
            final_error_support = str(support.get("stop_final_error_m") or "missing")
            status = "present" if defined_support == "present" and final_error_support in {"present", "approximate"} else "approximate"
        statuses.append(status)
        online_evidence.append(
            {
                "stage4_dir": str(stage4_dir),
                "exists": bool(payload),
                "measurement_support": support,
                "summary": {
                    "stop_target_defined_rate": summary.get("stop_target_defined_rate"),
                    "explicit_target_events": summary.get("explicit_target_events"),
                    "heuristic_target_events": summary.get("heuristic_target_events"),
                    "mean_stop_final_error_m": summary.get("mean_stop_final_error_m"),
                },
                "status": status,
            }
        )
    return _contract_status(statuses), {
        "frozen_replay_stage3c": frozen_evidence,
        "current_replay_stage3c": current_evidence,
        "online_stage4": online_evidence,
    }


def _latency_status(frozen_stage3c_dirs: list[Path], current_stage3c_dirs: list[Path], stage4_dirs: list[Path]) -> tuple[str, dict[str, Any]]:
    frozen_evidence: list[dict[str, Any]] = []
    current_evidence: list[dict[str, Any]] = []
    online_evidence: list[dict[str, Any]] = []
    statuses: list[str] = []
    for stage3c_dir in frozen_stage3c_dirs:
        rows = load_stage3c_jsonl(stage3c_dir / "planner_control_latency.jsonl")
        has_planner = any(row.get("planner_execution_latency_ms") is not None for row in rows)
        has_decision = any(row.get("decision_to_execution_latency_ms") is not None for row in rows)
        status = "present" if has_planner and has_decision else ("approximate" if rows else "missing")
        statuses.append(status)
        frozen_evidence.append(
            {
                "stage3c_dir": str(stage3c_dir),
                "row_count": len(rows),
                "has_planner_execution_latency": has_planner,
                "has_decision_to_execution_latency": has_decision,
                "status": status,
            }
        )
    for stage3c_dir in current_stage3c_dirs:
        rows = load_stage3c_jsonl(stage3c_dir / "planner_control_latency.jsonl")
        has_planner = any(row.get("planner_execution_latency_ms") is not None for row in rows)
        has_decision = any(row.get("decision_to_execution_latency_ms") is not None for row in rows)
        status = "present" if has_planner and has_decision else ("approximate" if rows else "missing")
        statuses.append(status)
        current_evidence.append(
            {
                "stage3c_dir": str(stage3c_dir),
                "row_count": len(rows),
                "has_planner_execution_latency": has_planner,
                "has_decision_to_execution_latency": has_decision,
                "status": status,
            }
        )
    for stage4_dir in stage4_dirs:
        rows = load_stage3c_jsonl(stage4_dir / "planner_control_latency.jsonl")
        has_planner = any(row.get("planner_execution_latency_ms") is not None for row in rows)
        has_decision = any(row.get("decision_to_execution_latency_ms") is not None for row in rows)
        has_total = any(row.get("total_loop_latency_ms") is not None for row in rows)
        status = "present" if has_planner and has_decision and has_total else ("approximate" if rows else "missing")
        statuses.append(status)
        online_evidence.append(
            {
                "stage4_dir": str(stage4_dir),
                "row_count": len(rows),
                "has_planner_execution_latency": has_planner,
                "has_decision_to_execution_latency": has_decision,
                "has_total_loop_latency": has_total,
                "status": status,
            }
        )
    return _contract_status(statuses), {"frozen_replay_stage3c": frozen_evidence, "current_replay_stage3c": current_evidence, "online_stage4": online_evidence}


def _fallback_status(frozen_stage3c_dirs: list[Path], current_stage3c_dirs: list[Path], stage4_dirs: list[Path]) -> tuple[str, dict[str, Any]]:
    frozen_evidence: list[dict[str, Any]] = []
    current_evidence: list[dict[str, Any]] = []
    online_evidence: list[dict[str, Any]] = []
    statuses: list[str] = []
    for stage3c_dir in frozen_stage3c_dirs:
        rows = load_stage3c_jsonl(stage3c_dir / "fallback_events.jsonl")
        activated = [row for row in rows if row.get("fallback_activated")]
        reasons = [row for row in activated if str(row.get("fallback_reason") or "").strip()]
        status = "present" if activated and len(reasons) == len(activated) else ("approximate" if rows else "missing")
        statuses.append(status)
        frozen_evidence.append(
            {
                "stage3c_dir": str(stage3c_dir),
                "row_count": len(rows),
                "activated_events": len(activated),
                "reason_covered_events": len(reasons),
                "status": status,
            }
        )
    for stage3c_dir in current_stage3c_dirs:
        rows = load_stage3c_jsonl(stage3c_dir / "fallback_events.jsonl")
        activated = [row for row in rows if row.get("fallback_activated")]
        reasons = [row for row in activated if str(row.get("fallback_reason") or "").strip()]
        status = "present" if activated and len(reasons) == len(activated) else ("approximate" if rows else "missing")
        statuses.append(status)
        current_evidence.append(
            {
                "stage3c_dir": str(stage3c_dir),
                "row_count": len(rows),
                "activated_events": len(activated),
                "reason_covered_events": len(reasons),
                "status": status,
            }
        )
    for stage4_dir in stage4_dirs:
        rows = load_stage3c_jsonl(stage4_dir / "fallback_events.jsonl")
        activated = [row for row in rows if row.get("fallback_activated")]
        reasons = [row for row in activated if str(row.get("fallback_reason") or "").strip()]
        has_takeover = any(row.get("takeover_latency_ms") is not None for row in activated)
        status = (
            "present"
            if activated and len(reasons) == len(activated) and has_takeover
            else ("approximate" if rows else "missing")
        )
        statuses.append(status)
        online_evidence.append(
            {
                "stage4_dir": str(stage4_dir),
                "row_count": len(rows),
                "activated_events": len(activated),
                "reason_covered_events": len(reasons),
                "has_takeover_latency": has_takeover,
                "status": status,
            }
        )
    return _contract_status(statuses), {"frozen_replay_stage3c": frozen_evidence, "current_replay_stage3c": current_evidence, "online_stage4": online_evidence}


def run_stage6_preflight_contract_audit(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_config = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    frozen_stage3c_dirs = _candidate_replay_stage3c_dirs(repo_root, benchmark_config)
    current_stage3c_dirs = _candidate_current_replay_stage3c_dirs(repo_root)
    stage4_dirs = _candidate_online_stage4_dirs(repo_root)

    for stage3c_dir in frozen_stage3c_dirs:
        _backfill_stage3c_dir(stage3c_dir)
    for stage3c_dir in current_stage3c_dirs:
        _backfill_stage3c_dir(stage3c_dir)
    for stage4_dir in stage4_dirs:
        _backfill_stage4_dir(stage4_dir)

    stop_target_contract, stop_target_evidence = _stop_target_status(frozen_stage3c_dirs, current_stage3c_dirs, stage4_dirs)
    latency_contract, latency_evidence = _latency_status(frozen_stage3c_dirs, current_stage3c_dirs, stage4_dirs)
    fallback_contract, fallback_evidence = _fallback_status(frozen_stage3c_dirs, current_stage3c_dirs, stage4_dirs)

    payload = {
        "schema_version": "stage6_preflight_contract_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stop_target_contract": stop_target_contract,
        "latency_contract": latency_contract,
        "fallback_contract": fallback_contract,
        "artifact_support": {
            "frozen_replay_stage3c_dirs": [str(path) for path in frozen_stage3c_dirs],
            "current_replay_stage3c_dirs": [str(path) for path in current_stage3c_dirs],
            "online_stage4_dirs": [str(path) for path in stage4_dirs],
            "stop_target": stop_target_evidence,
            "latency": latency_evidence,
            "fallback": fallback_evidence,
        },
        "notes": [
            "stop_target is considered approximate whenever final stop error is only proxy-backed rather than explicit for every stop source.",
            "latency is considered approximate when replay has direct planner timing but online stage breakdown is absent or partial.",
            "fallback is considered approximate when events exist but takeover latency or reason coverage is incomplete.",
        ],
    }
    dump_json(repo_root / "benchmark" / "stage6_preflight_contract_audit.json", payload)
    return payload
