from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.io import dump_json, load_json, load_yaml, resolve_repo_path  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_stage5a_gate_result_path() -> Path:
    return REPO_ROOT / "benchmark" / "stage5a_gate_result.json"


def _stage5a_run_map(stage5a_gate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = stage5a_gate.get("benchmark_runs") or []
    return {
        str(run.get("set")): run
        for run in runs
        if isinstance(run, dict) and run.get("set")
    }


def _report_dir_from_stage5a_gate(stage5a_gate: dict[str, Any], set_name: str) -> Path | None:
    run_map = _stage5a_run_map(stage5a_gate)
    entry = run_map.get(set_name) or {}
    report_dir = entry.get("report_dir")
    return Path(report_dir) if report_dir else None


def _case_warning_ids(benchmark_report: dict[str, Any]) -> list[str]:
    ready_cases = [c for c in (benchmark_report.get("cases") or []) if c.get("case_status") == "ready"]
    return [
        str(case.get("scenario_id"))
        for case in ready_cases
        if case.get("soft_warnings") or case.get("hard_failures")
    ]


def _stage5a_scenario_replay_clean(stage5a_gate: dict[str, Any]) -> bool:
    if str(stage5a_gate.get("benchmark_mode")) != "scenario_replay":
        return False
    if str(stage5a_gate.get("gate_driver")) != "frozen_corpus + scenario_replay":
        return False

    for run in stage5a_gate.get("benchmark_runs") or []:
        diagnostics = run.get("scenario_replay_diagnostics") or {}
        if run.get("requested_mode") != "scenario_replay":
            return False
        if diagnostics.get("skipped_cases") or diagnostics.get("runtime_failed_cases"):
            return False
    return True


def run_readiness(
    config_path: Path,
    output_path: Path,
    stage5a_gate_result_path: str | None = None,
    report_dir: str | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    metric_registry = load_yaml(resolve_repo_path(REPO_ROOT, cfg["metric_registry"]))
    stage5a_gate_path = Path(stage5a_gate_result_path) if stage5a_gate_result_path else _default_stage5a_gate_result_path()
    stage5a_gate = load_json(stage5a_gate_path, {}) or {}
    if not stage5a_gate:
        raise FileNotFoundError(f"No Stage 5A gate result found at {stage5a_gate_path}")

    core_report_dir = Path(report_dir) if report_dir else _report_dir_from_stage5a_gate(stage5a_gate, "stage5a_core_golden")
    failure_report_dir = _report_dir_from_stage5a_gate(stage5a_gate, "stage5a_failure_replay")
    if core_report_dir is None:
        raise FileNotFoundError("Stage 5A core report directory missing from stage5a_gate_result.json")

    core_gate = load_json(core_report_dir / "gate_result.json", {}) or {}
    core_bench = load_json(core_report_dir / "benchmark_report.json", {}) or {}
    failure_gate = load_json(failure_report_dir / "gate_result.json", {}) or {} if failure_report_dir else {}
    failure_bench = load_json(failure_report_dir / "benchmark_report.json", {}) or {} if failure_report_dir else {}

    metrics = metric_registry.get("metrics", {})
    hard_gate = sorted([name for name, meta in metrics.items() if meta.get("gate_usage") == "hard_gate"])
    soft_guardrail = sorted([name for name, meta in metrics.items() if meta.get("gate_usage") == "soft_guardrail"])

    ready_with_warnings = sorted(
        set(
            _case_warning_ids(core_bench)
            + _case_warning_ids(failure_bench)
        )
    )
    run_map = _stage5a_run_map(stage5a_gate)
    required_stage5a_sets = {"stage5a_core_golden", "stage5a_failure_replay"}
    missing_stage5a_sets = sorted(required_stage5a_sets - set(run_map))
    stage5a_gate_failed_cases = sorted(set(stage5a_gate.get("failed_cases") or []))
    stage5a_gate_warnings = sorted(set(stage5a_gate.get("warnings") or []))

    stage5a_ready = (
        stage5a_gate.get("overall_status") == "pass"
        and _stage5a_scenario_replay_clean(stage5a_gate)
        and not missing_stage5a_sets
        and not stage5a_gate_failed_cases
        and not stage5a_gate_warnings
        and core_gate.get("overall_status") == "pass"
        and not core_gate.get("failed_cases")
        and not core_gate.get("warning_cases")
        and not core_gate.get("semantic_degraded_cases")
        and (not failure_gate or (
            failure_gate.get("overall_status") == "pass"
            and not failure_gate.get("failed_cases")
            and not failure_gate.get("warning_cases")
            and not failure_gate.get("semantic_degraded_cases")
        ))
        and not ready_with_warnings
    )

    payload = {
        "benchmark_version": cfg.get("version"),
        "generated_at": _utc_now(),
        "evaluated_stage5a_gate_result_path": str(stage5a_gate_path),
        "evaluated_report_dir": str(core_report_dir),
        "evaluated_report_dirs": {
            "stage5a_core_golden": str(core_report_dir),
            "stage5a_failure_replay": str(failure_report_dir) if failure_report_dir else None,
        },
        "stage5a_gate_readiness": "ready" if stage5a_ready else "conditional",
        "stage5a_gate_status": stage5a_gate.get("overall_status"),
        "stage5a_gate_driver": stage5a_gate.get("gate_driver"),
        "stage5a_benchmark_mode": stage5a_gate.get("benchmark_mode"),
        "threshold_authoritative_source": stage5a_gate.get("threshold_authoritative_source"),
        "hard_gate_metrics": hard_gate,
        "soft_guardrail_metrics": soft_guardrail,
        "ready_case_warnings": ready_with_warnings,
        "failed_stage5a_cases": stage5a_gate_failed_cases,
        "warning_stage5a_cases": stage5a_gate_warnings,
        "missing_stage5a_sets": missing_stage5a_sets,
        "not_recommended_hard_gate_cases": [
            c.get("scenario_id")
            for c in (core_bench.get("cases") or [])
            if c.get("case_status") in {"provisional", "planned"}
        ],
        "gate_notes": [
            "Stage 5A readiness must be derived from benchmark/stage5a_gate_result.json, not a generic golden_* replay report.",
            "Scenario replay is the authoritative Stage 5A gate driver after frozen corpus materialization.",
            "Keep provisional/planned cases out of hard fail path.",
        ],
    }
    dump_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark readiness-gate recommendation.")
    parser.add_argument("--config", default=str(REPO_ROOT / "benchmark" / "benchmark_v1.yaml"))
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--stage5a-gate-result", default=str(_default_stage5a_gate_result_path()))
    parser.add_argument("--output", default=str(REPO_ROOT / "benchmark" / "reports" / "benchmark_readiness_gate.json"))
    args = parser.parse_args()

    payload = run_readiness(
        Path(args.config),
        Path(args.output),
        args.stage5a_gate_result,
        args.report_dir,
    )
    print(json.dumps({
        "stage5a_gate_readiness": payload["stage5a_gate_readiness"],
        "evaluated_report_dir": payload["evaluated_report_dir"],
        "evaluated_stage5a_gate_result_path": payload["evaluated_stage5a_gate_result_path"],
        "output": args.output,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
