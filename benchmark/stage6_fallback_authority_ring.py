from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import yaml

from scripts.run_system_benchmark_e2e import run_e2e_benchmark

from .carla_runtime import restart_carla_with_retry
from .io import dump_json, load_jsonl, load_yaml, resolve_repo_path


FALLBACK_RING_CASES = [
    "ml_right_positive_core",
    "stop_follow_ambiguity_core",
    "arbitration_stop_during_prepare_right",
    "fr_lc_commit_no_permission",
]

STAGE4_PROFILE_OVERRIDES = {
    "ml_right_positive_core": "stage4_online_external_watch_fallback_probe_low_memory",
    "stop_follow_ambiguity_core": "stage4_online_external_watch_stop_aligned_fallback_probe_low_memory",
    "arbitration_stop_during_prepare_right": "stage4_online_external_watch_stop_aligned_fallback_probe_arbitration_low_memory",
    "fr_lc_commit_no_permission": "stage4_online_external_watch_fallback_probe_low_memory",
}


INFRA_ERROR_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "infra_carla_rpc_timeout",
        (
            "client.get_world() not ready",
            "time-out of 10000ms while waiting for the simulator",
            "did not become ready",
        ),
    ),
    (
        "infra_carla_native_crash",
        (
            "exit_code: 3221226505",
            "exit_code: -1073740791",
            "exception_access_violation",
        ),
    ),
    (
        "infra_carla_connection_failure",
        (
            "connection refused",
            "failed to connect",
            "no process is listening on port",
        ),
    ),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _materialize_runtime_config(
    repo_root: Path,
    *,
    root: Path,
    case_id: str,
    stage4_profile: str,
) -> Path:
    benchmark_config = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    runtime_config = deepcopy(benchmark_config)
    cases_dir = resolve_repo_path(repo_root, benchmark_config.get("cases_dir")) or (repo_root / "benchmark" / "cases")
    runtime_cases_dir = root / "cases"
    runtime_sets_dir = root / "sets"
    runtime_cases_dir.mkdir(parents=True, exist_ok=True)
    runtime_sets_dir.mkdir(parents=True, exist_ok=True)

    case_payload = load_yaml(cases_dir / f"{case_id}.yaml")
    bundle_session_dir = (
        repo_root
        / "benchmark"
        / "frozen_corpus"
        / "v1"
        / "cases"
        / case_id
        / "stage1_frozen"
        / "session"
    )
    supported_modes = list(dict.fromkeys([*(case_payload.get("supported_benchmark_modes") or []), "online_e2e"]))
    case_payload["supported_benchmark_modes"] = supported_modes
    case_payload["benchmark_mode"] = "online_e2e"
    case_payload["stage_profiles"] = dict(case_payload.get("stage_profiles") or {})
    case_payload["stage_profiles"]["stage4"] = stage4_profile
    case_payload["scenario_recipe"] = dict(case_payload.get("scenario_recipe") or {})
    case_payload["scenario_recipe"]["stage1_session_dir"] = str(bundle_session_dir)
    case_payload["source"] = dict(case_payload.get("source") or {})
    case_payload["source"]["stage1_session_dir"] = str(bundle_session_dir)
    case_payload["notes"] = (
        str(case_payload.get("notes") or "").rstrip()
        + "\nStage 6 fallback authority ring override: low-memory external_watch fallback probe."
    ).strip()
    _write_yaml(runtime_cases_dir / f"{case_id}.yaml", case_payload)

    set_name = f"stage6_fallback_authority_{case_id}"
    set_path = runtime_sets_dir / f"{set_name}.yaml"
    _write_yaml(
        set_path,
        {
            "set_name": set_name,
            "description": f"Single-case fallback authority probe for {case_id}.",
            "cases": [case_id],
        },
    )
    runtime_config["cases_dir"] = str(runtime_cases_dir)
    runtime_config["sets"] = dict(runtime_config.get("sets") or {})
    runtime_config["sets"][set_name] = str(set_path)
    config_path = root / "benchmark_runtime_config.yaml"
    _write_yaml(config_path, runtime_config)
    return config_path


def _case_probe_payload(case_id: str, online_report_dir: Path, online_result: dict[str, Any]) -> dict[str, Any]:
    stage4_dir = online_report_dir / "case_runs" / case_id / "stage4"
    fallback_rows = load_jsonl(stage4_dir / "fallback_events.jsonl")
    latency_rows = load_jsonl(stage4_dir / "planner_control_latency.jsonl")
    activated_rows = [row for row in fallback_rows if bool(row.get("fallback_activated"))]
    probe_row = activated_rows[0] if activated_rows else {}
    probe_tick = probe_row.get("tick_id")
    planner_latency_ms = next(
        (
            row.get("planner_execution_latency_ms")
            for row in latency_rows
            if probe_tick is not None and int(row.get("tick_id", -1)) == int(probe_tick)
        ),
        None,
    )
    notes = list(probe_row.get("notes") or [])
    return {
        "case_id": case_id,
        "online_report_dir": str(online_report_dir),
        "trigger_type": "hard_timeout_fault_injection",
        "online_e2e_overall_status": online_result.get("overall_status"),
        "fallback_activated": bool(probe_row.get("fallback_activated")),
        "fallback_reason": probe_row.get("fallback_reason"),
        "takeover_latency_ms": probe_row.get("takeover_latency_ms"),
        "planner_execution_latency_ms": planner_latency_ms,
        "outcome": "success" if bool(probe_row.get("fallback_success")) else "failed_or_missing",
        "missing_when_required": bool(probe_row.get("missing_when_required")),
        "notes": notes,
        "artifact_paths": {
            "fallback_events": str(stage4_dir / "fallback_events.jsonl"),
            "planner_control_latency": str(stage4_dir / "planner_control_latency.jsonl"),
        },
    }


def _read_stage4_log_excerpt(stage4_log: Path) -> tuple[str | None, str | None]:
    if not stage4_log.exists():
        return None, None
    text = stage4_log.read_text(encoding="utf-8", errors="ignore")
    lower = text.lower()
    for failure_class, patterns in INFRA_ERROR_PATTERNS:
        for pattern in patterns:
            if pattern in lower:
                return failure_class, pattern
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return None, (lines[-1] if lines else None)


def run_stage6_fallback_authority_ring(
    repo_root: str | Path,
    *,
    carla_root: str | Path = "D:/carla",
    carla_port: int = 2000,
    retry_attempts: int = 1,
    case_startup_cooldown_seconds: float = 8.0,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    report_root = repo_root / "benchmark" / "reports" / f"stage6_fallback_authority_ring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_root.mkdir(parents=True, exist_ok=True)
    carla_root = Path(carla_root)

    case_payloads: list[dict[str, Any]] = []
    for case_id in FALLBACK_RING_CASES:
        attempt_payloads: list[dict[str, Any]] = []
        final_payload: dict[str, Any] | None = None
        for attempt_index in range(1, int(retry_attempts) + 2):
            restart_carla_with_retry(carla_root=carla_root, port=int(carla_port))
            time.sleep(float(case_startup_cooldown_seconds))
            case_root = report_root / case_id / f"attempt_{attempt_index}"
            config_path = _materialize_runtime_config(
                repo_root,
                root=case_root,
                case_id=case_id,
                stage4_profile=STAGE4_PROFILE_OVERRIDES[case_id],
            )
            online_report_dir = case_root / "online_e2e"
            online_result = run_e2e_benchmark(
                config_path=config_path,
                set_name=f"stage6_fallback_authority_{case_id}",
                benchmark_mode="online_e2e",
                execute_stages=True,
                report_dir=online_report_dir,
            )
            payload = _case_probe_payload(case_id, online_report_dir, online_result)
            stage4_log = online_report_dir / "case_runs" / case_id / "logs" / "stage4.log"
            failure_class, failure_reason = _read_stage4_log_excerpt(stage4_log)
            retryable_infra = bool(failure_class and payload["outcome"] == "failed_or_missing")
            attempt_payloads.append(
                {
                    "attempt_index": int(attempt_index),
                    "online_e2e_overall_status": payload.get("online_e2e_overall_status"),
                    "fallback_activated": payload.get("fallback_activated"),
                    "outcome": payload.get("outcome"),
                    "failure_class": failure_class,
                    "failure_reason": failure_reason,
                    "retryable_infra": retryable_infra,
                    "online_report_dir": str(online_report_dir),
                }
            )
            final_payload = payload
            if payload["outcome"] == "success":
                break
            if not retryable_infra or attempt_index > int(retry_attempts):
                break

        assert final_payload is not None
        final_payload["attempt_count"] = len(attempt_payloads)
        final_payload["attempts"] = attempt_payloads
        final_payload["recovered_after_retry"] = bool(
            len(attempt_payloads) > 1 and str(final_payload.get("outcome")) == "success"
        )
        case_payloads.append(final_payload)

    activated_count = sum(1 for row in case_payloads if bool(row.get("fallback_activated")))
    success_count = sum(1 for row in case_payloads if str(row.get("outcome")) == "success")
    latency_values = [
        float(row["takeover_latency_ms"])
        for row in case_payloads
        if row.get("takeover_latency_ms") is not None
    ]
    ready = (
        len(case_payloads) >= 2
        and activated_count == len(case_payloads)
        and success_count == len(case_payloads)
        and all(bool(str(row.get("fallback_reason") or "").strip()) for row in case_payloads)
        and all(not bool(row.get("missing_when_required")) for row in case_payloads)
    )
    report = {
        "schema_version": "stage6_fallback_authority_ring_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "cases": case_payloads,
        "aggregate": {
            "probe_case_count": len(case_payloads),
            "activated_case_count": activated_count,
            "success_case_count": success_count,
            "mean_takeover_latency_ms": (sum(latency_values) / len(latency_values)) if latency_values else None,
            "max_takeover_latency_ms": max(latency_values) if latency_values else None,
            "ready": ready,
            "coverage_scope": "multi_case_fault_injection" if len(case_payloads) >= 2 else "single_case_fault_injection",
        },
        "notes": [
            "This ring expands fallback evidence beyond the original single-case probe by forcing hard-timeout fallback activation on multiple promoted-ring cases.",
            "The ring remains proposal-only and does not transfer control authority to shadow MPC.",
            "Infrastructure retries are reserved for cold-start simulator failures; they do not mask missing fallback activation once the case reaches Stage 4 execution.",
        ],
    }
    result = {
        "schema_version": "stage6_fallback_authority_ring_result_v1",
        "generated_at_utc": report["generated_at_utc"],
        "overall_status": "ready" if ready else "not_ready",
        "activated_case_count": activated_count,
        "probe_case_count": len(case_payloads),
        "coverage_scope": report["aggregate"]["coverage_scope"],
        "report_path": str(repo_root / "benchmark" / "stage6_fallback_authority_ring_report.json"),
        "remaining_blockers": [] if ready else ["fallback_probe_activation_missing_or_incomplete"],
    }
    dump_json(repo_root / "benchmark" / "stage6_fallback_authority_ring_report.json", report)
    dump_json(repo_root / "benchmark" / "stage6_fallback_authority_ring_result.json", result)
    return {
        "report": report,
        "result": result,
    }
