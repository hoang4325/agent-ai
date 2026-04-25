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


ORGANIC_RING_CASES = [
    "stop_follow_ambiguity_core",
    "arbitration_stop_during_prepare_right",
]

ORGANIC_ALIGNMENT_BASE_PROFILE = "stage4_online_external_watch_shadow_smoke_extended_low_memory"

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


def _stage_profile_path_by_name(repo_root: Path, profile_name: str) -> Path:
    stage_profiles_dir = repo_root / "benchmark" / "stage_profiles"
    direct_path = stage_profiles_dir / f"{profile_name}.yaml"
    if direct_path.exists():
        return direct_path
    for path in sorted(stage_profiles_dir.glob("*.yaml")):
        payload = load_yaml(path)
        if str(payload.get("name") or path.stem) == str(profile_name):
            return path
    raise FileNotFoundError(f"Stage4 profile {profile_name} not found in {stage_profiles_dir}")


def _sample_index_from_name(sample_name: str | None) -> int | None:
    if not sample_name:
        return None
    prefix = "sample_"
    if not str(sample_name).startswith(prefix):
        return None
    suffix = str(sample_name)[len(prefix):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _infer_organic_alignment(repo_root: Path, case_id: str) -> dict[str, Any] | None:
    behavior_timeline_path = (
        repo_root
        / "benchmark"
        / "frozen_corpus"
        / "v1"
        / "cases"
        / case_id
        / "reference_outputs"
        / "stage3b"
        / "behavior_timeline_v2.jsonl"
    )
    if not behavior_timeline_path.exists():
        return None

    lane_change_behaviors = {
        "prepare_lane_change_left",
        "prepare_lane_change_right",
        "commit_lane_change_left",
        "commit_lane_change_right",
    }
    lane_change_segment_start: int | None = None
    lane_change_segment_start_sample: str | None = None
    last_lane_change_seq: int | None = None
    last_lane_change_sample: str | None = None

    with behavior_timeline_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = yaml.safe_load(line)
            sample_name = str(row.get("sample_name") or "")
            sequence_index = _sample_index_from_name(sample_name)
            if sequence_index is None:
                continue
            behavior = str(((row.get("behavior_request_v2") or {}).get("requested_behavior")) or "")

            if behavior in lane_change_behaviors:
                if lane_change_segment_start is None or (
                    last_lane_change_seq is not None and sequence_index - last_lane_change_seq > 2
                ):
                    lane_change_segment_start = sequence_index
                    lane_change_segment_start_sample = sample_name
                last_lane_change_seq = sequence_index
                last_lane_change_sample = sample_name
                continue

            if (
                behavior == "stop_before_obstacle"
                and lane_change_segment_start is not None
                and last_lane_change_seq is not None
                and 0 <= sequence_index - last_lane_change_seq <= 4
            ):
                start_index = lane_change_segment_start
                tick_span = max(12, min(20, sequence_index - start_index + 4))
                return {
                    "start_sequence_index": int(start_index),
                    "start_sample_name": lane_change_segment_start_sample or f"sample_{start_index:06d}",
                    "trigger_sequence_index": int(sequence_index),
                    "trigger_sample_name": sample_name,
                    "trigger_behavior": behavior,
                    "last_lane_change_sample_name": last_lane_change_sample,
                    "num_ticks": int(tick_span),
                    "alignment_reason": "lane_change_to_stop_transition",
                    "reference_behavior_timeline": str(behavior_timeline_path),
                }

            if last_lane_change_seq is not None and sequence_index - last_lane_change_seq > 4:
                lane_change_segment_start = None
                lane_change_segment_start_sample = None
                last_lane_change_seq = None
                last_lane_change_sample = None

    return None


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
    organic_alignment: dict[str, Any] | None = None,
) -> Path:
    benchmark_config = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    runtime_config = deepcopy(benchmark_config)
    cases_dir = resolve_repo_path(repo_root, benchmark_config.get("cases_dir")) or (repo_root / "benchmark" / "cases")
    runtime_cases_dir = root / "cases"
    runtime_sets_dir = root / "sets"
    runtime_stage_profiles_dir = root / "stage_profiles"
    runtime_cases_dir.mkdir(parents=True, exist_ok=True)
    runtime_sets_dir.mkdir(parents=True, exist_ok=True)
    runtime_stage_profiles_dir.mkdir(parents=True, exist_ok=True)

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
    case_payload["scenario_recipe"] = dict(case_payload.get("scenario_recipe") or {})
    case_payload["scenario_recipe"]["stage1_session_dir"] = str(bundle_session_dir)
    case_payload["source"] = dict(case_payload.get("source") or {})
    case_payload["source"]["stage1_session_dir"] = str(bundle_session_dir)

    base_profile_name = ORGANIC_ALIGNMENT_BASE_PROFILE if organic_alignment else stage4_profile
    base_profile_path = _stage_profile_path_by_name(repo_root, base_profile_name)
    base_profile_payload = load_yaml(base_profile_path)
    runtime_stage4_profile_name = f"stage6_organic_fallback_{case_id}_stage4"
    base_profile_payload["name"] = runtime_stage4_profile_name
    base_profile_payload["args"] = dict(base_profile_payload.get("args") or {})
    if organic_alignment:
        base_profile_payload["args"]["--external-watch-start-sequence-index"] = str(
            organic_alignment["start_sequence_index"]
        )
        base_profile_payload["args"]["--external-watch-start-sample-name"] = str(
            organic_alignment["start_sample_name"]
        )
        base_profile_payload["args"]["--external-watch-end-sequence-index"] = str(
            organic_alignment["start_sequence_index"] + organic_alignment["num_ticks"] - 1
        )
        base_profile_payload["args"]["--warmup-ticks"] = "0"
        base_profile_payload["args"]["--num-ticks"] = str(organic_alignment["num_ticks"])
        base_profile_payload["args"]["--external-watch-max-payloads-per-poll"] = "1"
    _write_yaml(runtime_stage_profiles_dir / f"{runtime_stage4_profile_name}.yaml", base_profile_payload)
    case_payload["stage_profiles"]["stage4"] = runtime_stage4_profile_name

    note_suffix = "Stage 6 organic fallback ring override: low-memory external_watch organic fallback observation only."
    if organic_alignment:
        note_suffix += (
            f" Organic window aligned to {organic_alignment['start_sample_name']} -> "
            f"{organic_alignment['trigger_sample_name']} ({organic_alignment['alignment_reason']})."
        )
    case_payload["notes"] = (str(case_payload.get("notes") or "").rstrip() + "\n" + note_suffix).strip()
    _write_yaml(runtime_cases_dir / f"{case_id}.yaml", case_payload)

    set_name = f"stage6_organic_fallback_{case_id}"
    set_path = runtime_sets_dir / f"{set_name}.yaml"
    _write_yaml(
        set_path,
        {
            "set_name": set_name,
            "description": f"Single-case organic fallback observation for {case_id}.",
            "cases": [case_id],
        },
    )
    runtime_config["cases_dir"] = str(runtime_cases_dir)
    runtime_config["stage_profiles_dir"] = str(runtime_stage_profiles_dir)
    runtime_config["sets"] = dict(runtime_config.get("sets") or {})
    runtime_config["sets"][set_name] = str(set_path)
    config_path = root / "benchmark_runtime_config.yaml"
    _write_yaml(config_path, runtime_config)
    return config_path


def _case_probe_payload(case_id: str, online_report_dir: Path, online_result: dict[str, Any]) -> dict[str, Any]:
    stage4_dir = online_report_dir / "case_runs" / case_id / "stage4"
    fallback_rows = load_jsonl(stage4_dir / "fallback_events.jsonl")
    tick_rows = load_jsonl(stage4_dir / "online_tick_record.jsonl")
    tick_fallback_rows = [
        {
            "tick_id": row.get("tick_id"),
            "request_frame": row.get("behavior_frame_id"),
            "sample_name": None,
            "carla_frame": row.get("frame_id"),
            "timestamp": row.get("timestamp"),
            "requested_behavior": row.get("selected_behavior"),
            "executing_behavior": row.get("fallback_action"),
            "planner_status": row.get("execution_status"),
            "fallback_required": row.get("fallback_required"),
            "fallback_activated": row.get("fallback_activated"),
            "fallback_reason": row.get("fallback_reason"),
            "fallback_source": row.get("fallback_source"),
            "fallback_mode": row.get("fallback_mode"),
            "fallback_action": row.get("fallback_action"),
            "takeover_latency_ms": row.get("fallback_takeover_latency_ms"),
            "fallback_success": row.get("fallback_success"),
            "missing_when_required": bool(row.get("fallback_required") and not row.get("fallback_activated")),
            "notes": list(row.get("notes") or []),
        }
        for row in tick_rows
    ]
    authoritative_source = "fallback_events.jsonl"
    if sum(1 for row in tick_fallback_rows if bool(row.get("fallback_activated"))) > sum(
        1 for row in fallback_rows if bool(row.get("fallback_activated"))
    ):
        fallback_rows = tick_fallback_rows
        authoritative_source = "online_tick_record.jsonl"
    activated_rows = [row for row in fallback_rows if bool(row.get("fallback_activated"))]
    organic_rows = [
        row
        for row in activated_rows
        if "fallback_probe:" not in str(row.get("fallback_reason") or "")
    ]
    successful_rows = [row for row in organic_rows if bool(row.get("fallback_success"))]
    probe_row = successful_rows[0] if successful_rows else (organic_rows[0] if organic_rows else {})
    return {
        "case_id": case_id,
        "online_report_dir": str(online_report_dir),
        "online_e2e_overall_status": online_result.get("overall_status"),
        "fallback_event_count": len(fallback_rows),
        "fallback_required_observed": any(bool(row.get("fallback_required")) for row in fallback_rows),
        "fallback_activated": bool(probe_row.get("fallback_activated")),
        "activation_count": len(organic_rows),
        "fallback_reason": probe_row.get("fallback_reason"),
        "takeover_latency_ms": probe_row.get("takeover_latency_ms"),
        "outcome": "success" if bool(probe_row.get("fallback_success")) else "not_observed",
        "missing_when_required": bool(probe_row.get("missing_when_required")),
        "notes": list(probe_row.get("notes") or []) + [f"authoritative_source={authoritative_source}"],
        "artifact_paths": {
            "fallback_events": str(stage4_dir / "fallback_events.jsonl"),
            "online_tick_record": str(stage4_dir / "online_tick_record.jsonl"),
        },
    }


def run_stage6_organic_fallback_ring(
    repo_root: str | Path,
    *,
    case_ids: list[str] | None = None,
    carla_root: str | Path = "D:/carla",
    carla_port: int = 2000,
    retry_attempts: int = 1,
    case_startup_cooldown_seconds: float = 8.0,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    report_root = repo_root / "benchmark" / "reports" / f"stage6_organic_fallback_ring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_root.mkdir(parents=True, exist_ok=True)
    carla_root = Path(carla_root)
    resolved_case_ids = list(case_ids or ORGANIC_RING_CASES)

    case_payloads: list[dict[str, Any]] = []
    for case_id in resolved_case_ids:
        attempt_payloads: list[dict[str, Any]] = []
        final_payload: dict[str, Any] | None = None
        organic_alignment = _infer_organic_alignment(repo_root, case_id)
        for attempt_index in range(1, int(retry_attempts) + 2):
            restart_carla_with_retry(carla_root=carla_root, port=int(carla_port))
            time.sleep(float(case_startup_cooldown_seconds))
            case_root = report_root / case_id / f"attempt_{attempt_index}"
            config_path = _materialize_runtime_config(
                repo_root,
                root=case_root,
                case_id=case_id,
                stage4_profile=LOW_MEMORY_STAGE4_PROFILE_OVERRIDES[case_id],
                organic_alignment=organic_alignment,
            )
            online_report_dir = case_root / "online_e2e"
            online_result = run_e2e_benchmark(
                config_path=config_path,
                set_name=f"stage6_organic_fallback_{case_id}",
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
        final_payload["organic_alignment"] = organic_alignment
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
        "schema_version": "stage6_organic_fallback_ring_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "probe_case_ids": resolved_case_ids,
        "cases": case_payloads,
        "aggregate": {
            "probe_case_count": len(case_payloads),
            "activated_case_count": activated_count,
            "fallback_required_case_count": required_count,
            "success_case_count": success_count,
            "mean_takeover_latency_ms": (sum(latency_values) / len(latency_values)) if latency_values else None,
            "max_takeover_latency_ms": max(latency_values) if latency_values else None,
            "ready": ready,
            "coverage_scope": "promoted_ring_organic_candidate_observation",
        },
        "notes": [
            "This ring targets promoted-ring cases whose reference Stage 3C traces already show organic fallback activation without timeout or route-uncertainty perturbation.",
            "It is narrower than the full promoted six-case ring by design; the goal is to establish real organic runtime evidence before considering broader authority-transfer work.",
        ],
    }
    result = {
        "schema_version": "stage6_organic_fallback_ring_result_v1",
        "generated_at_utc": report["generated_at_utc"],
        "overall_status": "ready" if ready else "not_ready",
        "activated_case_count": activated_count,
        "probe_case_count": len(case_payloads),
        "probe_case_ids": resolved_case_ids,
        "coverage_scope": report["aggregate"]["coverage_scope"],
        "report_path": str(repo_root / "benchmark" / "stage6_organic_fallback_ring_report.json"),
        "remaining_blockers": [] if ready else ["organic_fallback_activation_not_observed_multi_case"],
    }
    dump_json(repo_root / "benchmark" / "stage6_organic_fallback_ring_report.json", report)
    dump_json(repo_root / "benchmark" / "stage6_organic_fallback_ring_result.json", result)
    return {
        "report": report,
        "result": result,
    }
