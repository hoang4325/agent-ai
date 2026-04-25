"""
run_stage5b_gate.py
===================
Entrypoint to run the Stage 5B planner / execution quality gate:
1. Build Stage 5B metric pack and set files
2. Refresh frozen corpus
3. Audit artifact support for planner-quality metrics
4. Run scenario_replay on Stage 5B core / failure sets
5. Re-evaluate current-run bundles with Stage 5B runtime case overlays
6. Emit stage5b_gate_result.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.frozen_corpus_builder import build_frozen_corpus
from benchmark.io import dump_json, load_json, load_yaml, resolve_repo_path
from benchmark.runner_core import run_benchmark
from benchmark.stage5b_artifact_audit import run_stage5b_artifact_audit
from benchmark.stage5b_failure_replay_builder import build_stage5b_sets
from benchmark.stage5b_metric_pack import build_stage5b_metric_pack
from scripts.run_system_benchmark_e2e import run_e2e_benchmark


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _load_stage5b_case_sets(repo_root: Path) -> dict[str, list[str]]:
    benchmark_cfg = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    result: dict[str, list[str]] = {}
    for set_name in ["stage5b_core_golden", "stage5b_failure_replay"]:
        set_path = resolve_repo_path(repo_root, benchmark_cfg["sets"][set_name])
        payload = load_yaml(set_path)
        result[set_name] = list(payload.get("cases", []))
    return result


def _scenario_replay_diagnostics(result: dict[str, Any], required_cases: list[str]) -> dict[str, Any]:
    rows = list(result.get("mode_resolution") or [])
    row_by_case = {str(row.get("scenario_id")): row for row in rows}
    skipped_cases = []
    runtime_failed_cases = []
    successful_cases = []
    notes: list[str] = []

    for case_id in required_cases:
        row = row_by_case.get(case_id)
        if not row:
            skipped_cases.append(case_id)
            notes.append(f"scenario_replay_missing_mode_resolution:{case_id}")
            continue
        status = str(row.get("status"))
        if status == "success":
            successful_cases.append(case_id)
        elif status == "skipped":
            skipped_cases.append(case_id)
            notes.append(f"scenario_replay_skipped:{case_id}:{row.get('reason')}")
        elif status == "failed":
            runtime_failed_cases.append(case_id)
            notes.append(f"scenario_replay_runtime_failed:{case_id}:{row.get('reason')}")

    return {
        "required_cases": required_cases,
        "successful_cases": successful_cases,
        "skipped_cases": skipped_cases,
        "runtime_failed_cases": runtime_failed_cases,
        "notes": notes,
    }


def _apply_stage5b_overrides(case_spec: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    updated = dict(case_spec)
    mandatory = list(dict.fromkeys([*(updated.get("mandatory_artifacts") or []), *(override.get("mandatory_artifacts_add") or [])]))
    updated["mandatory_artifacts"] = mandatory
    if override.get("primary_metrics"):
        updated["primary_metrics"] = list(override["primary_metrics"])
    if override.get("secondary_metrics"):
        updated["secondary_metrics"] = list(override["secondary_metrics"])
    thresholds = dict(updated.get("metric_thresholds") or {})
    thresholds.update(override.get("metric_thresholds") or {})
    updated["metric_thresholds"] = thresholds
    updated["stage5b_runtime_overlay"] = {
        "enabled": True,
        "mandatory_artifacts_add": list(override.get("mandatory_artifacts_add") or []),
    }
    return updated


def _materialize_stage5b_runtime_config(
    *,
    e2e_report_root: Path,
    set_name: str,
    case_overrides: dict[str, Any],
) -> Path:
    base_runtime_cfg_path = e2e_report_root / "benchmark_runtime_config.yaml"
    base_runtime_cfg = load_yaml(base_runtime_cfg_path)
    base_cases_dir = e2e_report_root / "cases_runtime"
    overlay_cases_dir = e2e_report_root / "stage5b_cases_runtime"
    overlay_sets_dir = e2e_report_root / "stage5b_sets_runtime"
    overlay_cases_dir.mkdir(parents=True, exist_ok=True)
    overlay_sets_dir.mkdir(parents=True, exist_ok=True)

    set_cases = list(case_overrides.keys())
    for case_id in set_cases:
        case_spec = load_yaml(base_cases_dir / f"{case_id}.yaml")
        case_spec = _apply_stage5b_overrides(case_spec, case_overrides[case_id])
        _write_yaml(overlay_cases_dir / f"{case_id}.yaml", case_spec)

    set_payload = {
        "set_name": set_name,
        "description": f"Stage 5B runtime materialized set for {set_name}",
        "cases": set_cases,
    }
    overlay_set_path = overlay_sets_dir / f"{set_name}.yaml"
    _write_yaml(overlay_set_path, set_payload)

    runtime_cfg = dict(base_runtime_cfg)
    runtime_cfg["cases_dir"] = str(overlay_cases_dir)
    runtime_cfg["sets"] = dict(runtime_cfg.get("sets") or {})
    runtime_cfg["sets"][set_name] = str(overlay_set_path)
    runtime_cfg["stage5b_runtime_overlay"] = True
    runtime_cfg_path = e2e_report_root / f"stage5b_runtime_config_{set_name}.yaml"
    _write_yaml(runtime_cfg_path, runtime_cfg)
    return runtime_cfg_path


def _collect_failed_metrics(benchmark_report: dict[str, Any]) -> list[str]:
    failed_metrics: list[str] = []
    for case in benchmark_report.get("cases") or []:
        for key in ["hard_failures", "soft_warnings"]:
            for item in case.get(key) or []:
                name = str(item).split(":", 1)[0]
                if name and not name.startswith("missing_mandatory_artifact"):
                    failed_metrics.append(name)
    return sorted(set(failed_metrics))


def _evaluate_stage5b_current_run(
    *,
    e2e_result: dict[str, Any],
    set_name: str,
    case_overrides: dict[str, Any],
) -> dict[str, Any]:
    e2e_report_root = Path(str(e2e_result["report_dir"])).parent
    runtime_cfg_path = _materialize_stage5b_runtime_config(
        e2e_report_root=e2e_report_root,
        set_name=set_name,
        case_overrides=case_overrides,
    )
    eval_result = run_benchmark(
        repo_root=REPO_ROOT,
        config_path=runtime_cfg_path,
        set_name=set_name,
        report_dir=e2e_report_root / "stage5b_evaluation",
    )
    benchmark_report = load_json(Path(eval_result["report_dir"]) / "benchmark_report.json", default={}) or {}
    gate_result = load_json(Path(eval_result["report_dir"]) / "gate_result.json", default={}) or {}
    return {
        "runtime_config_path": str(runtime_cfg_path),
        "report_dir": eval_result["report_dir"],
        "gate_result": gate_result,
        "failed_metrics": _collect_failed_metrics(benchmark_report),
        "benchmark_report": benchmark_report,
    }


def _merge_gate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    failed_cases: list[str] = []
    warning_cases: list[str] = []
    failed_metrics: list[str] = []
    notes: list[str] = []
    for result in results:
        stage5b_eval = result.get("stage5b_evaluation") or {}
        gate = stage5b_eval.get("gate_result") or {}
        failed_cases.extend(gate.get("failed_cases") or [])
        warning_cases.extend(gate.get("warning_cases") or [])
        failed_metrics.extend(stage5b_eval.get("failed_metrics") or [])
        diagnostics = result.get("scenario_replay_diagnostics") or {}
        failed_cases.extend(diagnostics.get("skipped_cases") or [])
        failed_cases.extend(diagnostics.get("runtime_failed_cases") or [])
        notes.extend(diagnostics.get("notes") or [])

    overall = "pass"
    if failed_cases:
        overall = "fail"
    elif warning_cases:
        overall = "partial"
    return {
        "overall_status": overall,
        "failed_cases": sorted(set(failed_cases)),
        "warning_cases": sorted(set(warning_cases)),
        "failed_metrics": sorted(set(failed_metrics)),
        "notes": notes,
    }


def run_stage5b_gate(repo_root: str | Path | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root or REPO_ROOT)

    print("=" * 72)
    print("  STAGE 5B GATE PIPELINE")
    print("=" * 72)

    print("\n[Phase 1] Building Stage 5B metric pack...")
    metric_pack = build_stage5b_metric_pack(repo_root)
    print(f"  -> {len(metric_pack['hard_gate_metrics'])} hard gate metrics")
    print(f"  -> {len(metric_pack['soft_guardrail_metrics'])} soft guardrail metrics")
    print(f"  -> {len(metric_pack['diagnostic_metrics'])} diagnostic metrics")

    print("\n[Phase 2] Building Stage 5B set files...")
    build_stage5b_sets(repo_root)
    stage5b_sets = _load_stage5b_case_sets(repo_root)
    print(f"  -> stage5b_core_golden: {stage5b_sets['stage5b_core_golden']}")
    print(f"  -> stage5b_failure_replay: {stage5b_sets['stage5b_failure_replay']}")

    benchmark_cfg = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    frozen_root = resolve_repo_path(
        repo_root,
        ((benchmark_cfg.get("frozen_corpus") or {}).get("root")) or "benchmark/frozen_corpus/v1",
    )
    if frozen_root is None:
        raise ValueError("frozen corpus root is missing from benchmark config")

    print("\n[Phase 3] Refreshing frozen corpus...")
    frozen_build = build_frozen_corpus(
        repo_root=repo_root,
        config_path="benchmark/benchmark_v1.yaml",
        output_root=frozen_root,
    )
    print(f"  -> Frozen corpus refreshed at {frozen_root}")
    print(f"  -> Readiness counts: {frozen_build['readiness']['summary']['readiness_counts']}")

    print("\n[Phase 4] Auditing Stage 5B artifact support...")
    artifact_audit = run_stage5b_artifact_audit(repo_root)
    print(f"  -> Artifact support: {artifact_audit['artifact_support']}")

    benchmark_runs: list[dict[str, Any]] = []

    print("\n[Phase 5] Running stage5b_core_golden via scenario_replay...")
    core_result = run_e2e_benchmark(
        config_path=repo_root / "benchmark" / "benchmark_v1.yaml",
        set_name="stage5b_core_golden",
        benchmark_mode="scenario_replay",
        execute_stages=True,
    )
    core_result["scenario_replay_diagnostics"] = _scenario_replay_diagnostics(
        core_result,
        stage5b_sets["stage5b_core_golden"],
    )
    core_result["stage5b_evaluation"] = _evaluate_stage5b_current_run(
        e2e_result=core_result,
        set_name="stage5b_core_golden",
        case_overrides=(metric_pack["case_requirements"]["stage5b_core_golden"]["case_metric_overrides"]),
    )
    benchmark_runs.append(core_result)
    print(f"  -> Scenario replay status: {core_result.get('overall_status')}")
    print(f"  -> Stage 5B eval status: {core_result['stage5b_evaluation']['gate_result'].get('overall_status')}")

    print("\n[Phase 6] Running stage5b_failure_replay via scenario_replay...")
    failure_result = run_e2e_benchmark(
        config_path=repo_root / "benchmark" / "benchmark_v1.yaml",
        set_name="stage5b_failure_replay",
        benchmark_mode="scenario_replay",
        execute_stages=True,
    )
    failure_result["scenario_replay_diagnostics"] = _scenario_replay_diagnostics(
        failure_result,
        stage5b_sets["stage5b_failure_replay"],
    )
    failure_result["stage5b_evaluation"] = _evaluate_stage5b_current_run(
        e2e_result=failure_result,
        set_name="stage5b_failure_replay",
        case_overrides=(metric_pack["case_requirements"]["stage5b_failure_replay"]["case_metric_overrides"]),
    )
    benchmark_runs.append(failure_result)
    print(f"  -> Scenario replay status: {failure_result.get('overall_status')}")
    print(f"  -> Stage 5B eval status: {failure_result['stage5b_evaluation']['gate_result'].get('overall_status')}")

    print("\n[Phase 7] Computing final Stage 5B gate verdict...")
    merged_gate = _merge_gate_results(benchmark_runs)
    gate_output = {
        "schema_version": "stage5b_gate_result_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage": "5B",
        "benchmark_mode": "scenario_replay",
        "gate_driver": "frozen_corpus + scenario_replay + stage5b_runtime_overlay",
        "overall_status": merged_gate["overall_status"],
        "failed_metrics": merged_gate["failed_metrics"],
        "failed_cases": merged_gate["failed_cases"],
        "warnings": merged_gate["warning_cases"],
        "notes": merged_gate["notes"],
        "threshold_authoritative_source": {
            "primary": "stage5b_runtime_case_spec.metric_thresholds",
            "fallback": "benchmark_v1.default_metric_thresholds",
            "runner_resolution": "benchmark.runner_core._case_metric_thresholds",
        },
        "artifact_audit_path": str(repo_root / "benchmark" / "stage5b_artifact_audit.json"),
        "metric_pack_path": str(repo_root / "benchmark" / "stage5b_metric_pack.json"),
        "frozen_corpus_summary": frozen_build["readiness"]["summary"],
        "benchmark_runs": [
            {
                "set": run.get("set"),
                "requested_mode": run.get("requested_mode"),
                "scenario_replay_status": run.get("overall_status"),
                "stage5b_eval_status": ((run.get("stage5b_evaluation") or {}).get("gate_result") or {}).get("overall_status"),
                "failed_cases": ((run.get("stage5b_evaluation") or {}).get("gate_result") or {}).get("failed_cases", []),
                "warning_cases": ((run.get("stage5b_evaluation") or {}).get("gate_result") or {}).get("warning_cases", []),
                "failed_metrics": (run.get("stage5b_evaluation") or {}).get("failed_metrics", []),
                "scenario_replay_report_dir": run.get("report_dir"),
                "stage5b_evaluation_report_dir": (run.get("stage5b_evaluation") or {}).get("report_dir"),
                "scenario_replay_diagnostics": run.get("scenario_replay_diagnostics"),
            }
            for run in benchmark_runs
        ],
    }

    output_path = repo_root / "benchmark" / "stage5b_gate_result.json"
    dump_json(output_path, gate_output)

    print("\n" + "=" * 72)
    print(f"  STAGE 5B GATE VERDICT: {merged_gate['overall_status'].upper()}")
    print("=" * 72)
    if merged_gate["failed_cases"]:
        print(f"  Failed cases: {merged_gate['failed_cases']}")
    if merged_gate["failed_metrics"]:
        print(f"  Failed metrics: {merged_gate['failed_metrics']}")
    if merged_gate["warning_cases"]:
        print(f"  Warning cases: {merged_gate['warning_cases']}")
    print(f"\n  Gate result written to: {output_path}")
    return gate_output


if __name__ == "__main__":
    result = run_stage5b_gate()
    sys.exit(0 if result["overall_status"] == "pass" else 1)
