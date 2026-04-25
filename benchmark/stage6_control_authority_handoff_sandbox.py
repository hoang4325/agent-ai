from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import yaml

from scripts.run_system_benchmark_e2e import run_e2e_benchmark

from .carla_runtime import restart_carla_with_retry, wait_for_carla_runtime_stable
from .io import dump_json, load_json, load_yaml, resolve_repo_path


DEFAULT_SANDBOX_CASE_ID = "ml_right_positive_core"
NO_MOTION_STAGE4_PROFILE = "stage4_online_external_watch_shadow_authority_sandbox_low_memory"
BOUNDED_MOTION_STAGE4_PROFILE = "stage4_online_external_watch_shadow_authority_bounded_motion_low_memory"
TRANSFER_STAGE4_PROFILE = "stage4_online_external_watch_shadow_authority_transfer_low_memory"
FULL_AUTHORITY_STAGE4_PROFILE = "stage4_online_external_watch_shadow_authority_full_low_memory"
FULL_AUTHORITY_STOP_STAGE4_PROFILE = "stage4_online_external_watch_stop_aligned_shadow_authority_full_low_memory"

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


def _infer_infra_failure_from_exception(exc: Exception | None) -> tuple[str | None, str | None]:
    if exc is None:
        return None, None
    text = str(exc).strip()
    lower = text.lower()
    if (
        "client.get_world() not ready" in lower
        or "time-out of 10000ms while waiting for the simulator" in lower
        or "did not become ready" in lower
        or "runtime did not stabilize" in lower
    ):
        return "infra_carla_rpc_timeout", text
    if (
        "exit_code: 3221226505" in lower
        or "exit_code: -1073740791" in lower
        or "exception_access_violation" in lower
    ):
        return "infra_carla_native_crash", text
    if (
        "connection refused" in lower
        or "failed to connect" in lower
        or "no process is listening on port" in lower
    ):
        return "infra_carla_connection_failure", text
    return None, text


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
    note_suffix: str,
    set_name: str,
    description: str,
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
        + f"\n{note_suffix}"
    ).strip()
    _write_yaml(runtime_cases_dir / f"{case_id}.yaml", case_payload)

    set_path = runtime_sets_dir / f"{set_name}.yaml"
    _write_yaml(
        set_path,
        {
            "set_name": set_name,
            "description": description,
            "cases": [case_id],
        },
    )
    runtime_config["cases_dir"] = str(runtime_cases_dir)
    runtime_config["sets"] = dict(runtime_config.get("sets") or {})
    runtime_config["sets"][set_name] = str(set_path)
    config_path = root / "benchmark_runtime_config.yaml"
    _write_yaml(config_path, runtime_config)
    return config_path


def _run_control_authority_sandbox(
    repo_root: Path,
    *,
    case_id: str,
    stage4_profile: str,
    set_name: str,
    report_stem: str,
    result_stem: str,
    note_suffix: str,
    description: str,
    expected_mode: str,
    require_shadow_takeover: bool,
    report_notes: list[str],
    carla_root: str | Path,
    carla_port: int,
    retry_attempts: int = 1,
    case_startup_cooldown_seconds: float = 8.0,
    post_restart_stability_timeout_s: float | None = None,
    post_restart_stability_consecutive_successes: int = 3,
    post_restart_stability_poll_seconds: float = 1.0,
) -> dict[str, Any]:
    report_root = repo_root / "benchmark" / "reports" / f"{report_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_root.mkdir(parents=True, exist_ok=True)
    online_result: dict[str, Any] | None = None
    online_report_dir: Path | None = None
    sandbox_summary: dict[str, Any] = {}
    attempts: list[dict[str, Any]] = []
    for attempt_index in range(1, int(retry_attempts) + 2):
        restart_carla_with_retry(carla_root=Path(carla_root), port=int(carla_port))
        time.sleep(float(case_startup_cooldown_seconds))
        if post_restart_stability_timeout_s is not None and float(post_restart_stability_timeout_s) > 0.0:
            stability_probe = wait_for_carla_runtime_stable(
                carla_root=Path(carla_root),
                port=int(carla_port),
                timeout_s=float(post_restart_stability_timeout_s),
                consecutive_successes=int(post_restart_stability_consecutive_successes),
                poll_seconds=float(post_restart_stability_poll_seconds),
            )
            if str(stability_probe.get("status") or "") != "stable":
                attempts.append(
                    {
                        "attempt_index": int(attempt_index),
                        "online_e2e_overall_status": "preflight_failed",
                        "failure_class": "infra_carla_runtime_not_stable_after_cooldown",
                        "failure_reason": stability_probe.get("last_error") or stability_probe,
                        "retryable_infra": True,
                        "online_report_dir": None,
                        "preflight_probe_only": True,
                    }
                )
                if attempt_index <= int(retry_attempts):
                    continue
                online_result = {
                    "overall_status": "fail",
                    "error": (
                        "CARLA runtime did not remain stable after restart cooldown. "
                        f"probe={stability_probe}"
                    ),
                }
                online_report_dir = report_root / f"attempt_{attempt_index}" / "online_e2e"
                sandbox_summary = {}
                break
        attempt_root = report_root / f"attempt_{attempt_index}"
        config_path = _materialize_runtime_config(
            repo_root,
            root=attempt_root,
            case_id=case_id,
            stage4_profile=stage4_profile,
            note_suffix=note_suffix,
            set_name=set_name,
            description=description,
        )
        online_report_dir = attempt_root / "online_e2e"
        run_error: Exception | None = None
        try:
            online_result = run_e2e_benchmark(
                config_path=config_path,
                set_name=set_name,
                benchmark_mode="online_e2e",
                execute_stages=True,
                report_dir=online_report_dir,
            )
        except RuntimeError as exc:
            run_error = exc
            online_result = {
                "overall_status": "fail",
                "error": str(exc),
            }
        stage4_dir = online_report_dir / "case_runs" / case_id / "stage4"
        sandbox_summary = load_json(stage4_dir / "control_authority_sandbox_summary.json", default={}) or {}
        stage4_log = online_report_dir / "case_runs" / case_id / "logs" / "stage4.log"
        failure_class, failure_reason = _read_stage4_log_excerpt(stage4_log)
        if failure_class is None:
            failure_class, failure_reason = _infer_infra_failure_from_exception(run_error)
        retryable_infra = bool(failure_class and not sandbox_summary)
        attempts.append(
            {
                "attempt_index": int(attempt_index),
                "online_e2e_overall_status": (online_result or {}).get("overall_status"),
                "failure_class": failure_class,
                "failure_reason": failure_reason,
                "retryable_infra": retryable_infra,
                "online_report_dir": str(online_report_dir),
            }
        )
        if sandbox_summary:
            break
        if not retryable_infra or attempt_index > int(retry_attempts):
            break

    assert online_result is not None
    assert online_report_dir is not None
    stage4_dir = online_report_dir / "case_runs" / case_id / "stage4"
    ready = (
        str(online_result.get("overall_status")) in {"pass", "partial", "fail"}
        and str(sandbox_summary.get("sandbox_mode")) == expected_mode
        and bool(sandbox_summary.get("handoff_executed"))
        and bool(sandbox_summary.get("max_speed_within_limit"))
        and bool(sandbox_summary.get("actual_shadow_takeover_applied")) == bool(require_shadow_takeover)
        and bool(sandbox_summary.get("ready_for_sandbox_discussion"))
    )
    report = {
        "schema_version": "stage6_control_authority_handoff_sandbox_report_v2",
        "generated_at_utc": _utc_now_iso(),
        "case_id": case_id,
        "online_report_dir": str(online_report_dir),
        "online_e2e_overall_status": online_result.get("overall_status"),
        "sandbox_summary_path": str(stage4_dir / "control_authority_sandbox_summary.json"),
        "sandbox_trace_path": str(stage4_dir / "control_authority_sandbox_trace.jsonl"),
        "sandbox_summary": sandbox_summary,
        "expected_mode": expected_mode,
        "attempts": attempts,
        "recovered_after_retry": bool(len(attempts) > 1 and sandbox_summary),
        "ready": ready,
        "notes": list(report_notes),
    }
    result = {
        "schema_version": "stage6_control_authority_handoff_sandbox_result_v2",
        "generated_at_utc": report["generated_at_utc"],
        "overall_status": "ready" if ready else "not_ready",
        "case_id": case_id,
        "expected_mode": expected_mode,
        "report_path": str(repo_root / "benchmark" / f"{report_stem}.json"),
        "remaining_blockers": [] if ready else [f"{expected_mode}_sandbox_execution_or_guard_incomplete"],
    }
    dump_json(repo_root / "benchmark" / f"{report_stem}.json", report)
    dump_json(repo_root / "benchmark" / f"{result_stem}.json", result)
    return {
        "report": report,
        "result": result,
    }


def run_stage6_control_authority_handoff_sandbox(
    repo_root: str | Path,
    *,
    case_id: str = DEFAULT_SANDBOX_CASE_ID,
    carla_root: str | Path = "D:/carla",
    carla_port: int = 2000,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    return _run_control_authority_sandbox(
        repo_root,
        case_id=case_id,
        stage4_profile=NO_MOTION_STAGE4_PROFILE,
        set_name="stage6_control_authority_handoff_sandbox",
        report_stem="stage6_control_authority_handoff_sandbox_report",
        result_stem="stage6_control_authority_handoff_sandbox_result",
        note_suffix="Stage 6 control-authority sandbox override: no-motion handoff trace with baseline control locked.",
        description="Single-case no-motion control-authority handoff sandbox.",
        expected_mode="no_motion_handoff",
        require_shadow_takeover=False,
        report_notes=[
            "This sandbox executes the authority state-machine path while baseline remains the only actuator owner.",
            "Shadow proposals are visible to the sandbox state machine but never acquire control authority.",
        ],
        carla_root=carla_root,
        carla_port=int(carla_port),
    )


def run_stage6_control_authority_bounded_motion_sandbox(
    repo_root: str | Path,
    *,
    case_id: str = DEFAULT_SANDBOX_CASE_ID,
    carla_root: str | Path = "D:/carla",
    carla_port: int = 2000,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    return _run_control_authority_sandbox(
        repo_root,
        case_id=case_id,
        stage4_profile=BOUNDED_MOTION_STAGE4_PROFILE,
        set_name="stage6_control_authority_bounded_motion_sandbox",
        report_stem="stage6_control_authority_bounded_motion_sandbox_report",
        result_stem="stage6_control_authority_bounded_motion_sandbox_result",
        note_suffix="Stage 6 control-authority sandbox override: bounded-motion handoff trace with baseline control still locked.",
        description="Single-case bounded-motion control-authority handoff sandbox.",
        expected_mode="bounded_motion_handoff",
        require_shadow_takeover=False,
        report_notes=[
            "This sandbox executes the authority state-machine path while baseline remains the only actuator owner.",
            "Shadow proposals are visible to the sandbox state machine but never acquire control authority.",
        ],
        carla_root=carla_root,
        carla_port=int(carla_port),
    )


def run_stage6_control_authority_transfer_sandbox(
    repo_root: str | Path,
    *,
    case_id: str = DEFAULT_SANDBOX_CASE_ID,
    stage4_profile: str = TRANSFER_STAGE4_PROFILE,
    set_name: str = "stage6_control_authority_transfer_sandbox",
    report_stem: str = "stage6_control_authority_transfer_sandbox_report",
    result_stem: str = "stage6_control_authority_transfer_sandbox_result",
    note_suffix: str = "Stage 6 control-authority sandbox override: bounded-motion handoff with tightly bounded shadow actuator authority transfer.",
    description: str = "Single-case bounded-motion shadow authority transfer sandbox.",
    carla_root: str | Path = "D:/carla",
    carla_port: int = 2000,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    return _run_control_authority_sandbox(
        repo_root,
        case_id=case_id,
        stage4_profile=stage4_profile,
        set_name=set_name,
        report_stem=report_stem,
        result_stem=result_stem,
        note_suffix=note_suffix,
        description=description,
        expected_mode="bounded_motion_shadow_authority",
        require_shadow_takeover=True,
        report_notes=[
            "This sandbox applies a tightly bounded shadow-derived control for a short window on a single case.",
            "It is still a sandbox artifact and must not be treated as takeover rollout approval.",
        ],
        carla_root=carla_root,
        carla_port=int(carla_port),
        retry_attempts=2,
        case_startup_cooldown_seconds=12.0,
    )


def run_stage6_control_authority_full_authority_sandbox(
    repo_root: str | Path,
    *,
    case_id: str = DEFAULT_SANDBOX_CASE_ID,
    stage4_profile: str = FULL_AUTHORITY_STAGE4_PROFILE,
    set_name: str = "stage6_control_authority_full_authority_sandbox",
    report_stem: str = "stage6_control_authority_full_authority_sandbox_report",
    result_stem: str = "stage6_control_authority_full_authority_sandbox_result",
    note_suffix: str = "Stage 6 control-authority sandbox override: full-runtime shadow actuator authority after handoff without mandatory rollback.",
    description: str = "Single-case full-runtime shadow authority sandbox.",
    carla_root: str | Path = "D:/carla",
    carla_port: int = 2000,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    return _run_control_authority_sandbox(
        repo_root,
        case_id=case_id,
        stage4_profile=stage4_profile,
        set_name=set_name,
        report_stem=report_stem,
        result_stem=result_stem,
        note_suffix=note_suffix,
        description=description,
        expected_mode="full_shadow_authority",
        require_shadow_takeover=True,
        report_notes=[
            "This sandbox applies shadow-derived control after handoff for the full remaining runtime window.",
            "It is still a sandbox artifact and must not be treated as production takeover rollout approval.",
        ],
        carla_root=carla_root,
        carla_port=int(carla_port),
        retry_attempts=2,
        case_startup_cooldown_seconds=24.0,
        post_restart_stability_timeout_s=40.0,
        post_restart_stability_consecutive_successes=5,
        post_restart_stability_poll_seconds=2.0,
    )
