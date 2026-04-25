"""
run_stage6_shadow_gate.py
=========================
Stage 6 MPC shadow rollout entrypoint:
1. Verify Stage 6 shadow readiness is already green on the promoted subset
2. Run scenario_replay for the two-case shadow subset
3. Materialize MPC shadow proposal artifacts in each Stage 3C current-run bundle
4. Re-evaluate those current-run bundles with Stage 6 shadow metrics
5. Emit Stage 6 shadow contract audit / report / gate result
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.io import dump_json, load_json, load_yaml, resolve_repo_path
from benchmark.runner_core import run_benchmark
from benchmark.shadow_artifacts import generate_mpc_shadow_artifacts
from benchmark.stage6_shadow_contract_audit import run_stage6_shadow_contract_audit
from benchmark.stage6_shadow_metric_pack import build_stage6_shadow_metric_pack
from scripts.run_system_benchmark_e2e import run_e2e_benchmark


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _load_shadow_subset(repo_root: Path) -> list[str]:
    benchmark_cfg = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    set_path = resolve_repo_path(repo_root, benchmark_cfg["sets"]["stage6_shadow_golden"])
    payload = load_yaml(set_path)
    return list(payload.get("cases", []))


def _scenario_replay_diagnostics(result: dict[str, Any], required_cases: list[str]) -> dict[str, Any]:
    rows = list(result.get("mode_resolution") or [])
    row_by_case = {str(row.get("scenario_id")): row for row in rows}
    skipped_cases: list[str] = []
    runtime_failed_cases: list[str] = []
    successful_cases: list[str] = []
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
        else:
            runtime_failed_cases.append(case_id)
            notes.append(f"scenario_replay_runtime_failed:{case_id}:{row.get('reason')}")
    return {
        "required_cases": required_cases,
        "successful_cases": successful_cases,
        "skipped_cases": skipped_cases,
        "runtime_failed_cases": runtime_failed_cases,
        "notes": notes,
    }


def _apply_stage6_shadow_overrides(
    *,
    case_spec: dict[str, Any],
    override: dict[str, Any],
    case_run_dir: Path,
) -> dict[str, Any]:
    updated = dict(case_spec)
    updated_artifacts = dict(updated.get("artifacts") or {})
    updated_artifacts.update(
        {
            "stage3c_mpc_shadow_proposal": str(case_run_dir / "stage3c" / "mpc_shadow_proposal.jsonl"),
            "stage3c_mpc_shadow_summary": str(case_run_dir / "stage3c" / "mpc_shadow_summary.json"),
            "stage3c_shadow_baseline_comparison": str(case_run_dir / "stage3c" / "shadow_baseline_comparison.json"),
            "stage3c_shadow_disagreement_events": str(case_run_dir / "stage3c" / "shadow_disagreement_events.jsonl"),
            "stage3c_shadow_realized_outcome_trace": str(case_run_dir / "stage3c" / "shadow_realized_outcome_trace.jsonl"),
            "stage3c_shadow_realized_outcome_summary": str(case_run_dir / "stage3c" / "shadow_realized_outcome_summary.json"),
        }
    )
    updated["artifacts"] = updated_artifacts
    updated["mandatory_artifacts"] = list(
        dict.fromkeys([*(updated.get("mandatory_artifacts") or []), *(override.get("mandatory_artifacts_add") or [])])
    )
    if override.get("primary_metrics"):
        updated["primary_metrics"] = list(override["primary_metrics"])
    if override.get("secondary_metrics"):
        updated["secondary_metrics"] = list(override["secondary_metrics"])
    thresholds = dict(updated.get("metric_thresholds") or {})
    thresholds.update(override.get("metric_thresholds") or {})
    updated["metric_thresholds"] = thresholds
    updated["stage6_shadow_runtime_overlay"] = {
        "enabled": True,
        "subset_case": updated.get("scenario_id"),
    }
    return updated


def _materialize_stage6_shadow_runtime_config(
    *,
    e2e_report_root: Path,
    set_name: str,
    case_overrides: dict[str, Any],
) -> Path:
    base_runtime_cfg = load_yaml(e2e_report_root / "benchmark_runtime_config.yaml")
    base_cases_dir = e2e_report_root / "cases_runtime"
    overlay_cases_dir = e2e_report_root / "stage6_shadow_cases_runtime"
    overlay_sets_dir = e2e_report_root / "stage6_shadow_sets_runtime"
    overlay_cases_dir.mkdir(parents=True, exist_ok=True)
    overlay_sets_dir.mkdir(parents=True, exist_ok=True)

    set_cases = list(case_overrides.keys())
    for case_id in set_cases:
        case_spec = load_yaml(base_cases_dir / f"{case_id}.yaml")
        case_run_dir = e2e_report_root / "case_runs" / case_id
        case_spec = _apply_stage6_shadow_overrides(
            case_spec=case_spec,
            override=case_overrides[case_id],
            case_run_dir=case_run_dir,
        )
        _write_yaml(overlay_cases_dir / f"{case_id}.yaml", case_spec)

    overlay_set_path = overlay_sets_dir / f"{set_name}.yaml"
    _write_yaml(
        overlay_set_path,
        {
            "set_name": set_name,
            "description": f"Stage 6 shadow runtime materialized set for {set_name}",
            "cases": set_cases,
        },
    )

    runtime_cfg = dict(base_runtime_cfg)
    runtime_cfg["cases_dir"] = str(overlay_cases_dir)
    runtime_cfg["sets"] = dict(runtime_cfg.get("sets") or {})
    runtime_cfg["sets"][set_name] = str(overlay_set_path)
    runtime_cfg["stage6_shadow_runtime_overlay"] = True
    runtime_cfg_path = e2e_report_root / f"stage6_shadow_runtime_config_{set_name}.yaml"
    _write_yaml(runtime_cfg_path, runtime_cfg)
    return runtime_cfg_path


def _collect_failed_metrics(
    benchmark_report: dict[str, Any],
    *,
    diagnostic_metrics: list[str],
) -> list[str]:
    diagnostic_set = set(diagnostic_metrics)
    failed_metrics: list[str] = []
    for case in benchmark_report.get("cases") or []:
        case_metrics = case.get("metrics") or {}
        for metric_name, metric_result in case_metrics.items():
            if metric_name in diagnostic_set:
                continue
            if metric_result.get("threshold_passed") is False:
                failed_metrics.append(metric_name)
    return sorted(set(failed_metrics))


def _materialize_shadow_outputs(e2e_report_root: Path, case_ids: list[str]) -> dict[str, Any]:
    per_case: dict[str, Any] = {}
    for case_id in case_ids:
        stage3c_dir = e2e_report_root / "case_runs" / case_id / "stage3c"
        per_case[case_id] = generate_mpc_shadow_artifacts(stage3c_dir, case_id=case_id)
    return per_case


def _build_shadow_report(
    *,
    repo_root: Path,
    metric_pack: dict[str, Any],
    contract_audit: dict[str, Any],
    e2e_result: dict[str, Any],
    shadow_case_outputs: dict[str, Any],
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    cases = []
    for case_id, payload in shadow_case_outputs.items():
        summary = dict(payload.get("summary") or {})
        comparison = dict(payload.get("comparison") or {})
        cases.append(
            {
                "case_id": case_id,
                "shadow_summary": summary,
                "comparison": comparison,
                "proposal_rows": len(payload.get("proposal_rows") or []),
                "disagreement_rows": len(payload.get("disagreement_events") or []),
            }
        )
    report = {
        "schema_version": "stage6_shadow_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage": "6_shadow",
        "description": "MPC shadow-mode preflight rollout report on the promoted two-case subset.",
        "shadow_readiness_path": str(repo_root / "benchmark" / "stage6_shadow_readiness.json"),
        "shadow_metric_pack_path": str(repo_root / "benchmark" / "stage6_shadow_metric_pack.json"),
        "shadow_contract_audit_path": str(repo_root / "benchmark" / "stage6_shadow_contract_audit.json"),
        "scenario_replay_report_dir": e2e_result.get("report_dir"),
        "shadow_evaluation_report_dir": eval_result.get("report_dir"),
        "subset_cases": metric_pack.get("subset_cases"),
        "cases": cases,
        "contract_audit": contract_audit,
        "notes": [
            "Baseline remains the only execution authority.",
            "Shadow outputs are proposal-only and benchmarked without takeover.",
            "Current shadow rollout uses an OSQP-backed minimal MPC shadow solver; shadow quality deltas remain non-authoritative until a fuller MPC solver is integrated.",
        ],
    }
    dump_json(repo_root / "benchmark" / "stage6_shadow_report.json", report)
    return report


def run_stage6_shadow_gate(repo_root: str | Path | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root or REPO_ROOT)
    readiness = load_json(repo_root / "benchmark" / "stage6_shadow_readiness.json", default={}) or {}
    if str(readiness.get("status")) != "ready":
        raise RuntimeError(
            "Stage 6 shadow rollout is blocked because benchmark/stage6_shadow_readiness.json is not ready."
        )

    metric_pack = build_stage6_shadow_metric_pack(repo_root)
    required_cases = _load_shadow_subset(repo_root)

    e2e_result = run_e2e_benchmark(
        config_path=repo_root / "benchmark" / "benchmark_v1.yaml",
        set_name="stage6_shadow_golden",
        benchmark_mode="scenario_replay",
        execute_stages=True,
    )
    diagnostics = _scenario_replay_diagnostics(e2e_result, required_cases)

    e2e_report_root = Path(str(e2e_result["report_dir"])).parent
    shadow_case_outputs = _materialize_shadow_outputs(e2e_report_root, required_cases)
    contract_audit = run_stage6_shadow_contract_audit(
        repo_root,
        case_stage3c_dirs={
            case_id: e2e_report_root / "case_runs" / case_id / "stage3c"
            for case_id in required_cases
        },
    )

    runtime_cfg_path = _materialize_stage6_shadow_runtime_config(
        e2e_report_root=e2e_report_root,
        set_name="stage6_shadow_golden",
        case_overrides=metric_pack["case_requirements"]["stage6_shadow_golden"]["case_metric_overrides"],
    )
    eval_result = run_benchmark(
        repo_root=repo_root,
        config_path=runtime_cfg_path,
        set_name="stage6_shadow_golden",
        report_dir=e2e_report_root / "stage6_shadow_evaluation",
    )
    benchmark_report = load_json(Path(eval_result["report_dir"]) / "benchmark_report.json", default={}) or {}
    gate_result = load_json(Path(eval_result["report_dir"]) / "gate_result.json", default={}) or {}
    shadow_report = _build_shadow_report(
        repo_root=repo_root,
        metric_pack=metric_pack,
        contract_audit=contract_audit,
        e2e_result=e2e_result,
        shadow_case_outputs=shadow_case_outputs,
        eval_result=eval_result,
    )

    failed_cases = sorted(
        set((gate_result.get("failed_cases") or []) + diagnostics.get("skipped_cases", []) + diagnostics.get("runtime_failed_cases", []))
    )
    warning_cases = sorted(set(gate_result.get("warning_cases") or []))
    failed_metrics = _collect_failed_metrics(
        benchmark_report,
        diagnostic_metrics=[item["metric"] for item in metric_pack.get("diagnostic_metrics") or []],
    )
    overall_status = "pass"
    if failed_cases:
        overall_status = "fail"
    elif warning_cases:
        overall_status = "partial"

    gate_payload = {
        "schema_version": "stage6_shadow_gate_result_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage": "6_shadow",
        "benchmark_mode": "scenario_replay",
        "gate_driver": "frozen_corpus + scenario_replay + stage6_shadow_runtime_overlay",
        "overall_status": overall_status,
        "failed_metrics": failed_metrics,
        "failed_cases": failed_cases,
        "warnings": warning_cases,
        "notes": diagnostics.get("notes", []),
        "shadow_contract_audit_path": str(repo_root / "benchmark" / "stage6_shadow_contract_audit.json"),
        "shadow_metric_pack_path": str(repo_root / "benchmark" / "stage6_shadow_metric_pack.json"),
        "shadow_report_path": str(repo_root / "benchmark" / "stage6_shadow_report.json"),
        "scenario_replay_report_dir": e2e_result.get("report_dir"),
        "shadow_evaluation_report_dir": eval_result.get("report_dir"),
        "shadow_contract_status": contract_audit.get("shadow_output_contract"),
        "baseline_comparison_ready": contract_audit.get("baseline_comparison_ready"),
        "subset_cases": required_cases,
        "hard_gate_metrics": [item["metric"] for item in metric_pack.get("hard_gate_metrics") or []],
        "soft_guardrail_metrics": [item["metric"] for item in metric_pack.get("soft_guardrail_metrics") or []],
        "diagnostic_metrics": [item["metric"] for item in metric_pack.get("diagnostic_metrics") or []],
    }
    dump_json(repo_root / "benchmark" / "stage6_shadow_gate_result.json", gate_payload)
    return {
        "gate_result": gate_payload,
        "shadow_report": shadow_report,
        "scenario_replay_result": e2e_result,
        "shadow_evaluation": eval_result,
        "contract_audit": contract_audit,
        "metric_pack": metric_pack,
    }


if __name__ == "__main__":
    result = run_stage6_shadow_gate()
    print(result["gate_result"]["overall_status"])
    sys.exit(0 if result["gate_result"]["overall_status"] == "pass" else 1)
