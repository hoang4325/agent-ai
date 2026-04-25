from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.io import load_yaml, resolve_repo_path  # noqa: E402
from benchmark.contracts import default_trace_requirements, expected_stage_chain  # noqa: E402
from benchmark.frozen_corpus_support import overlay_case_with_frozen_corpus  # noqa: E402
from benchmark.orchestration import execute_case  # noqa: E402
from benchmark.provenance import TraceLogger, new_id  # noqa: E402
from benchmark.runner_core import run_benchmark  # noqa: E402
from benchmark.carla_runtime import ensure_carla_ready, restart_carla  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark with optional scenario_replay / online_e2e orchestration.")
    parser.add_argument("--config", default=str(REPO_ROOT / "benchmark" / "benchmark_v1.yaml"))
    parser.add_argument("--set", dest="set_name", required=True)
    parser.add_argument("--benchmark-mode", choices=["auto", "replay_regression", "scenario_replay", "online_e2e"], default="auto")
    parser.add_argument("--execute-stages", action="store_true", help="Actually execute stage runners for non-replay modes.")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--report-dir", default=None)
    return parser


def _load_case_spec(repo_root: Path, cases_dir: Path, case_id: str) -> dict[str, Any]:
    case_path = cases_dir / f"{case_id}.yaml"
    payload = load_yaml(case_path)
    payload["_case_path"] = str(case_path)
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _mode_for_case(case_spec: dict[str, Any], override_mode: str) -> str:
    if override_mode != "auto":
        return override_mode
    return str(case_spec.get("benchmark_mode", "replay_regression"))


def _apply_online_overrides(case_spec: dict[str, Any], modes_cfg: dict[str, Any]) -> None:
    overrides = (((modes_cfg.get("online_e2e_defaults") or {}).get("evaluation_case_overrides")) or {})
    for key in ["mandatory_artifacts", "primary_metrics", "secondary_metrics"]:
        if key not in overrides:
            continue
        existing = list(case_spec.get(key) or [])
        merged = list(dict.fromkeys([*existing, *(overrides.get(key) or [])]))
        case_spec[key] = merged
    metric_thresholds = dict(case_spec.get("metric_thresholds") or {})
    metric_thresholds.update(overrides.get("metric_thresholds") or {})
    case_spec["metric_thresholds"] = metric_thresholds


def run_e2e_benchmark(
    *,
    config_path: str | Path,
    set_name: str,
    benchmark_mode: str = "auto",
    execute_stages: bool = False,
    python_executable: str | None = None,
    report_dir: str | Path | None = None,
    restart_carla_between_online_cases: bool = False,
    carla_root: str | Path = "D:/carla",
    carla_port: int = 2000,
    carla_case_startup_cooldown_seconds: float = 8.0,
) -> dict[str, Any]:
    benchmark_cfg = load_yaml(Path(config_path))
    modes_cfg = load_yaml(resolve_repo_path(REPO_ROOT, benchmark_cfg.get("benchmark_modes", "benchmark/benchmark_modes.yaml")))
    cases_dir = resolve_repo_path(REPO_ROOT, benchmark_cfg["cases_dir"])
    set_path = resolve_repo_path(REPO_ROOT, benchmark_cfg["sets"][set_name])
    if cases_dir is None or set_path is None:
        raise ValueError("Invalid benchmark config: missing cases/set paths")

    set_payload = load_yaml(set_path)
    case_ids = list(set_payload.get("cases", []))

    benchmark_run_id = new_id("benchmark_run")
    default_report_root = REPO_ROOT / "benchmark" / "reports" / f"e2e_{set_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_root = Path(report_dir) if report_dir else default_report_root
    report_root.mkdir(parents=True, exist_ok=True)

    trace_logger = TraceLogger(
        benchmark_run_id=benchmark_run_id,
        trace_file=report_root / "benchmark_execution_trace.jsonl",
        contract_trace_file=report_root / "contract_trace.jsonl",
    )

    temp_cases_dir = report_root / "cases_runtime"
    temp_sets_dir = report_root / "sets_runtime"
    temp_cases_dir.mkdir(parents=True, exist_ok=True)
    temp_sets_dir.mkdir(parents=True, exist_ok=True)

    stage_profiles_dir = resolve_repo_path(REPO_ROOT, benchmark_cfg.get("stage_profiles_dir", "benchmark/stage_profiles"))
    if stage_profiles_dir is None:
        raise ValueError("stage_profiles_dir missing")

    runtime_case_ids: list[str] = []
    case_mode_rows: list[dict[str, Any]] = []

    for case_id in case_ids:
        case_spec = _load_case_spec(REPO_ROOT, cases_dir, case_id)
        mode = _mode_for_case(case_spec, benchmark_mode)
        if mode == "online_e2e" and restart_carla_between_online_cases:
            restart_carla(carla_root=Path(carla_root), port=int(carla_port))
            cooldown_seconds = max(0.0, float(carla_case_startup_cooldown_seconds))
            if cooldown_seconds > 0.0:
                time.sleep(cooldown_seconds)
        elif mode == "online_e2e":
            ensure_carla_ready(carla_root=Path(carla_root), port=int(carla_port))
        case_spec, overlay_status = overlay_case_with_frozen_corpus(
            repo_root=REPO_ROOT,
            benchmark_config=benchmark_cfg,
            case_spec=case_spec,
            benchmark_mode=mode,
        )
        if overlay_status:
            case_spec["frozen_corpus_overlay_status"] = overlay_status
        supported_modes = set(case_spec.get("supported_benchmark_modes") or [case_spec.get("benchmark_mode", "replay_regression")])
        if mode not in supported_modes:
            case_mode_rows.append({"scenario_id": case_id, "mode": mode, "status": "skipped", "reason": "unsupported_mode"})
            continue

        exec_result = execute_case(
            repo_root=REPO_ROOT,
            benchmark_run_id=benchmark_run_id,
            case_spec=case_spec,
            benchmark_mode=mode,
            run_root=report_root / "case_runs",
            python_executable=python_executable,
            execute_stages=bool(execute_stages),
            stage_profiles_dir=stage_profiles_dir,
            trace_logger=trace_logger,
        )

        if exec_result.status == "fail":
            case_mode_rows.append({"scenario_id": case_id, "mode": mode, "status": "failed", "reason": ";".join(exec_result.notes)})
            continue

        if mode == "online_e2e":
            _apply_online_overrides(case_spec, modes_cfg)

        if exec_result.artifacts:
            merged_artifacts = dict(case_spec.get("artifacts") or {})
            merged_artifacts.update(exec_result.artifacts)
            case_spec["artifacts"] = merged_artifacts
        if exec_result.source_updates:
            merged_source = dict(case_spec.get("source") or {})
            merged_source.update(exec_result.source_updates)
            case_spec["source"] = merged_source

        case_spec["benchmark_mode"] = mode
        case_spec["expected_stage_chain"] = expected_stage_chain(case_spec, mode)
        case_spec["trace_requirements"] = default_trace_requirements(case_spec, mode)
        case_spec["trace_requirements"]["benchmark_run_id"] = benchmark_run_id
        case_spec["trace_requirements"]["trace_id"] = exec_result.trace_id
        case_spec["trace_requirements"]["contract_trace_path"] = str(report_root / "contract_trace.jsonl")
        case_spec["orchestration_status"] = exec_result.status

        runtime_case_ids.append(case_id)
        case_mode_rows.append(
            {
                "scenario_id": case_id,
                "mode": mode,
                "status": exec_result.status,
                "trace_id": exec_result.trace_id,
                "artifact_provenance_path": exec_result.artifact_provenance_path,
                "stage_chain_result_path": exec_result.stage_chain_result_path,
            }
        )
        _write_yaml(temp_cases_dir / f"{case_id}.yaml", case_spec)

    runtime_set_path = temp_sets_dir / f"{set_name}.yaml"
    _write_yaml(
        runtime_set_path,
        {
            "set_name": f"{set_name}_runtime",
            "description": f"Runtime materialized set for {set_name}",
            "cases": runtime_case_ids,
        },
    )

    runtime_cfg = dict(benchmark_cfg)
    runtime_cfg["cases_dir"] = str(temp_cases_dir)
    runtime_cfg["sets"] = dict(runtime_cfg.get("sets") or {})
    runtime_cfg["sets"][set_name] = str(runtime_set_path)
    runtime_cfg_path = report_root / "benchmark_runtime_config.yaml"
    _write_yaml(runtime_cfg_path, runtime_cfg)

    if not runtime_case_ids:
        (report_root / "mode_resolution.json").write_text(json.dumps(case_mode_rows, indent=2) + "\n", encoding="utf-8")
        return {
            "benchmark_run_id": benchmark_run_id,
            "set": set_name,
            "requested_mode": benchmark_mode,
            "execute_stages": bool(execute_stages),
            "overall_status": "fail",
            "reason": "no_runtime_cases_materialized",
            "report_dir": str(report_root),
            "trace_file": str(report_root / "benchmark_execution_trace.jsonl"),
            "mode_resolution": case_mode_rows,
            "benchmark_result": None,
            "gate_result": {"overall_status": "fail", "failed_cases": list(case_ids)},
        }

    benchmark_result = run_benchmark(
        repo_root=REPO_ROOT,
        config_path=runtime_cfg_path,
        set_name=set_name,
        report_dir=report_root / "evaluation",
    )

    online_case_results = []
    for row in case_mode_rows:
        if row.get("mode") != "online_e2e" or row.get("status") != "success":
            continue
        online_case_results.append(
            {
                "case_id": row["scenario_id"],
                "benchmark_mode": "online_e2e",
                "stage_chain": expected_stage_chain(_load_case_spec(REPO_ROOT, temp_cases_dir, row["scenario_id"]), "online_e2e"),
                "run_status": row.get("status", "partial"),
                "evaluation_report_path": str(Path(benchmark_result["report_dir"]) / "benchmark_report.json"),
                "gate_result_path": str(Path(benchmark_result["report_dir"]) / "gate_result.json"),
                "trace_id": row.get("trace_id"),
                "artifact_provenance_path": row.get("artifact_provenance_path"),
                "stage_chain_result_path": row.get("stage_chain_result_path"),
                "contract_trace_path": str(report_root / "contract_trace.jsonl"),
            }
        )

    (report_root / "online_e2e_case_result.json").write_text(json.dumps(online_case_results, indent=2) + "\n", encoding="utf-8")
    (report_root / "mode_resolution.json").write_text(json.dumps(case_mode_rows, indent=2) + "\n", encoding="utf-8")
    (report_root / "benchmark_run_manifest.json").write_text(
        json.dumps(
            {
                "benchmark_run_id": benchmark_run_id,
                "set": set_name,
                "requested_mode": benchmark_mode,
                "execute_stages": bool(execute_stages),
                "report_dir": str(report_root),
                "runtime_config_path": str(runtime_cfg_path),
                "trace_file": str(report_root / "benchmark_execution_trace.jsonl"),
                "contract_trace_file": str(report_root / "contract_trace.jsonl"),
                "mode_resolution_path": str(report_root / "mode_resolution.json"),
                "online_case_result_path": str(report_root / "online_e2e_case_result.json"),
                "cases_materialized": runtime_case_ids,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "benchmark_run_id": benchmark_run_id,
        "set": set_name,
        "requested_mode": benchmark_mode,
        "execute_stages": bool(execute_stages),
        "overall_status": benchmark_result["gate_result"]["overall_status"],
        "report_dir": benchmark_result["report_dir"],
        "trace_file": str(report_root / "benchmark_execution_trace.jsonl"),
        "mode_resolution": case_mode_rows,
        "benchmark_result": benchmark_result,
        "gate_result": benchmark_result["gate_result"],
    }


def main() -> int:
    args = build_parser().parse_args()
    result = run_e2e_benchmark(
        config_path=args.config,
        set_name=args.set_name,
        benchmark_mode=args.benchmark_mode,
        execute_stages=bool(args.execute_stages),
        python_executable=args.python_executable,
        report_dir=args.report_dir,
    )
    console = {
        "benchmark_run_id": result["benchmark_run_id"],
        "set": result["set"],
        "requested_mode": result["requested_mode"],
        "execute_stages": result["execute_stages"],
        "overall_status": result["overall_status"],
        "report_dir": result["report_dir"],
        "trace_file": result["trace_file"],
    }
    print(json.dumps(console, indent=2))
    return 0 if result["overall_status"] in {"pass", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
