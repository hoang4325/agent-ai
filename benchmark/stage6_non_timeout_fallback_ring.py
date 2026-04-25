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
from .stage6_shadow_matrix import LOW_MEMORY_STAGE4_PROFILE_OVERRIDES


PROMOTED_RING_CASES = [
    "ml_right_positive_core",
    "stop_follow_ambiguity_core",
    "ml_left_positive_core",
    "junction_straight_core",
    "arbitration_stop_during_prepare_right",
    "fr_lc_commit_no_permission",
]

NON_TIMEOUT_PERTURBATION_CASES = {
    "ml_right_positive_core",
    "junction_straight_core",
    "fr_lc_commit_no_permission",
}

NON_TIMEOUT_STAGE4_PROFILE = "stage4_online_external_watch_fallback_route_uncertainty_low_memory"

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
        + "\nStage 6 non-timeout fallback ring override: low-memory external_watch route-uncertainty perturbation evidence."
    ).strip()
    _write_yaml(runtime_cases_dir / f"{case_id}.yaml", case_payload)

    set_name = f"stage6_non_timeout_fallback_{case_id}"
    set_path = runtime_sets_dir / f"{set_name}.yaml"
    _write_yaml(
        set_path,
        {
            "set_name": set_name,
            "description": f"Single-case non-timeout fallback evidence for {case_id}.",
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
    activated_rows = [row for row in fallback_rows if bool(row.get("fallback_activated"))]
    perturbation_rows = [
        row
        for row in activated_rows
        if "fallback_perturbation:route_uncertainty" in str(row.get("fallback_reason") or "")
    ]
    probe_row = perturbation_rows[0] if perturbation_rows else {}
    return {
        "case_id": case_id,
        "perturbation_enabled": bool(case_id in NON_TIMEOUT_PERTURBATION_CASES),
        "online_report_dir": str(online_report_dir),
        "online_e2e_overall_status": online_result.get("overall_status"),
        "fallback_event_count": len(fallback_rows),
        "fallback_required_observed": any(bool(row.get("fallback_required")) for row in fallback_rows),
        "fallback_activated": bool(probe_row.get("fallback_activated")),
        "activation_count": len(perturbation_rows),
        "fallback_reason": probe_row.get("fallback_reason"),
        "takeover_latency_ms": probe_row.get("takeover_latency_ms"),
        "outcome": "success" if bool(probe_row.get("fallback_success")) else "not_observed",
        "missing_when_required": bool(probe_row.get("missing_when_required")),
        "notes": list(probe_row.get("notes") or []),
        "artifact_paths": {
            "fallback_events": str(stage4_dir / "fallback_events.jsonl"),
        },
    }


def run_stage6_non_timeout_fallback_ring(
    repo_root: str | Path,
    *,
    carla_root: str | Path = "D:/carla",
    carla_port: int = 2000,
    retry_attempts: int = 1,
    case_startup_cooldown_seconds: float = 8.0,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    report_root = repo_root / "benchmark" / "reports" / f"stage6_non_timeout_fallback_ring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_root.mkdir(parents=True, exist_ok=True)
    carla_root = Path(carla_root)

    case_payloads: list[dict[str, Any]] = []
    for case_id in PROMOTED_RING_CASES:
        attempt_payloads: list[dict[str, Any]] = []
        final_payload: dict[str, Any] | None = None
        for attempt_index in range(1, int(retry_attempts) + 2):
            restart_carla_with_retry(carla_root=carla_root, port=int(carla_port))
            time.sleep(float(case_startup_cooldown_seconds))
            case_root = report_root / case_id / f"attempt_{attempt_index}"
            stage4_profile = (
                NON_TIMEOUT_STAGE4_PROFILE
                if case_id in NON_TIMEOUT_PERTURBATION_CASES
                else LOW_MEMORY_STAGE4_PROFILE_OVERRIDES[case_id]
            )
            config_path = _materialize_runtime_config(
                repo_root,
                root=case_root,
                case_id=case_id,
                stage4_profile=stage4_profile,
            )
            online_report_dir = case_root / "online_e2e"
            online_result = run_e2e_benchmark(
                config_path=config_path,
                set_name=f"stage6_non_timeout_fallback_{case_id}",
                benchmark_mode="online_e2e",
                execute_stages=True,
                report_dir=online_report_dir,
            )
            payload = _case_probe_payload(case_id, online_report_dir, online_result)
            stage4_log = online_report_dir / "case_runs" / case_id / "logs" / "stage4.log"
            failure_class, failure_reason = _read_stage4_log_excerpt(stage4_log)
            retryable_infra = bool(failure_class and payload["outcome"] == "not_observed")
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
    required_count = sum(1 for row in case_payloads if bool(row.get("fallback_required_observed")))
    success_count = sum(1 for row in case_payloads if str(row.get("outcome")) == "success")
    latency_values = [
        float(row["takeover_latency_ms"])
        for row in case_payloads
        if row.get("takeover_latency_ms") is not None
    ]
    ready = (
        len(case_payloads) >= 2
        and activated_count >= 2
        and success_count == activated_count
        and all(not bool(row.get("missing_when_required")) for row in case_payloads)
    )
    report = {
        "schema_version": "stage6_non_timeout_fallback_ring_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "cases": case_payloads,
        "aggregate": {
            "probe_case_count": len(case_payloads),
            "activated_case_count": activated_count,
            "fallback_required_case_count": required_count,
            "success_case_count": success_count,
            "mean_takeover_latency_ms": (sum(latency_values) / len(latency_values)) if latency_values else None,
            "max_takeover_latency_ms": max(latency_values) if latency_values else None,
            "ready": ready,
            "coverage_scope": "promoted_ring_non_timeout_route_uncertainty",
        },
        "notes": [
            "This ring uses a narrow non-timeout route-uncertainty perturbation on selected promoted-ring cases to exercise fallback downgrade behavior.",
            "It is stronger than timeout-only fault injection, but it is still not equivalent to fully organic fallback activation coverage.",
        ],
    }
    result = {
        "schema_version": "stage6_non_timeout_fallback_ring_result_v1",
        "generated_at_utc": report["generated_at_utc"],
        "overall_status": "ready" if ready else "not_ready",
        "activated_case_count": activated_count,
        "probe_case_count": len(case_payloads),
        "coverage_scope": report["aggregate"]["coverage_scope"],
        "report_path": str(repo_root / "benchmark" / "stage6_non_timeout_fallback_ring_report.json"),
        "remaining_blockers": [] if ready else ["non_timeout_fallback_activation_not_observed_multi_case"],
    }
    dump_json(repo_root / "benchmark" / "stage6_non_timeout_fallback_ring_report.json", report)
    dump_json(repo_root / "benchmark" / "stage6_non_timeout_fallback_ring_result.json", result)
    return {
        "report": report,
        "result": result,
    }
