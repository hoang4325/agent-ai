from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.run_system_benchmark_e2e import run_e2e_benchmark

from .io import dump_json, dump_jsonl, load_json, load_jsonl, load_yaml, resolve_repo_path


FALLBACK_PROBE_CASE = "ml_right_positive_core"


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _materialize_runtime_config(repo_root: Path, *, root: Path) -> Path:
    benchmark_config = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    runtime_config = deepcopy(benchmark_config)
    cases_dir = resolve_repo_path(repo_root, benchmark_config.get("cases_dir")) or (repo_root / "benchmark" / "cases")
    runtime_cases_dir = root / "cases"
    runtime_cases_dir.mkdir(parents=True, exist_ok=True)

    case_payload = load_yaml(cases_dir / f"{FALLBACK_PROBE_CASE}.yaml")
    bundle_session_dir = (
        repo_root
        / "benchmark"
        / "frozen_corpus"
        / "v1"
        / "cases"
        / FALLBACK_PROBE_CASE
        / "stage1_frozen"
        / "session"
    )
    case_payload["stage_profiles"] = dict(case_payload.get("stage_profiles") or {})
    case_payload["stage_profiles"]["stage4"] = "stage4_online_external_watch_fallback_probe"
    case_payload["scenario_recipe"] = dict(case_payload.get("scenario_recipe") or {})
    case_payload["scenario_recipe"]["stage1_session_dir"] = str(bundle_session_dir)
    case_payload["source"] = dict(case_payload.get("source") or {})
    case_payload["source"]["stage1_session_dir"] = str(bundle_session_dir)
    _write_yaml(runtime_cases_dir / f"{FALLBACK_PROBE_CASE}.yaml", case_payload)

    set_name = "stage4_fallback_probe_subset"
    set_path = root / "sets" / f"{set_name}.yaml"
    _write_yaml(
        set_path,
        {
            "set_name": set_name,
            "description": "Single-case online fallback activation probe with minimal hard-timeout injection.",
            "cases": [FALLBACK_PROBE_CASE],
        },
    )
    runtime_config["cases_dir"] = str(runtime_cases_dir)
    runtime_config["sets"] = dict(runtime_config.get("sets") or {})
    runtime_config["sets"][set_name] = str(set_path)
    config_path = root / "benchmark_runtime_config.yaml"
    _write_yaml(config_path, runtime_config)
    return config_path


def run_fallback_activation_probe(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    report_root = repo_root / "benchmark" / "reports" / f"stage4_fallback_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_root.mkdir(parents=True, exist_ok=True)
    config_path = _materialize_runtime_config(repo_root, root=report_root)

    online_report_dir = report_root / "online_e2e"
    online_result = run_e2e_benchmark(
        config_path=config_path,
        set_name="stage4_fallback_probe_subset",
        benchmark_mode="online_e2e",
        execute_stages=True,
        report_dir=online_report_dir,
    )

    stage4_dir = online_report_dir / "case_runs" / FALLBACK_PROBE_CASE / "stage4"
    fallback_rows = load_jsonl(stage4_dir / "fallback_events.jsonl")
    latency_rows = load_jsonl(stage4_dir / "planner_control_latency.jsonl")
    tick_rows = load_jsonl(stage4_dir / "online_tick_record.jsonl")
    activated_rows = [row for row in fallback_rows if bool(row.get("fallback_activated"))]
    activated_trace = [
        row for row in fallback_rows
        if bool(row.get("fallback_activated")) or any("fallback_probe:" in str(note) for note in (row.get("notes") or []))
    ]
    probe_row = activated_rows[0] if activated_rows else {}
    tick_lookup = {
        int(row.get("tick_id")): row
        for row in tick_rows
        if row.get("tick_id") is not None
    }
    probe_tick = probe_row.get("tick_id")
    tick_payload = tick_lookup.get(int(probe_tick)) if probe_tick is not None else {}

    probe_payload = {
        "schema_version": "stage6.fallback_activation_probe.v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "case_id": FALLBACK_PROBE_CASE,
        "report_root": str(online_report_dir),
        "trigger_type": "hard_timeout_fault_injection",
        "fallback_activated": bool(probe_row.get("fallback_activated")),
        "fallback_reason": probe_row.get("fallback_reason"),
        "takeover_latency_ms": probe_row.get("takeover_latency_ms"),
        "outcome": "success" if bool(probe_row.get("fallback_success")) else "failed_or_missing",
        "planner_execution_latency_ms": next(
            (
                row.get("planner_execution_latency_ms")
                for row in latency_rows
                if probe_tick is not None and int(row.get("tick_id", -1)) == int(probe_tick)
            ),
            None,
        ),
        "tick_notes": list(tick_payload.get("notes") or []),
        "online_e2e_overall_status": online_result.get("overall_status"),
    }
    dump_json(repo_root / "benchmark" / "fallback_activation_probe.json", probe_payload)
    dump_jsonl(repo_root / "benchmark" / "fallback_takeover_latency_trace.jsonl", activated_trace)
    return {
        "report_root": str(report_root),
        "online_result": online_result,
        "probe": probe_payload,
    }
