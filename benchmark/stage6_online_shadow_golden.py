from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .io import dump_json, load_json, load_yaml, resolve_repo_path
from .runner_core import run_benchmark
from .stage6_online_shadow_smoke import SHADOW_CASES, run_stage6_online_shadow_smoke


BASELINE_DIR = Path("benchmark") / "baselines" / "system_benchmark_v1_stage6_online_shadow"
BASELINE_METRICS = BASELINE_DIR / "baseline_metrics.json"
BASELINE_COMMIT = BASELINE_DIR / "baseline_commit.json"
BASELINE_DECISION = Path("benchmark") / "stage6_online_shadow_golden_baseline_decision.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _latest_pass_shadow_smoke_run(repo_root: Path) -> Path | None:
    reports_root = repo_root / "benchmark" / "reports"
    candidates = sorted([path for path in reports_root.glob("stage6_online_shadow_smoke_*") if path.is_dir()])
    passing: list[Path] = []
    for report_dir in candidates:
        gate_result = load_json(report_dir / "online_e2e" / "stage6_online_shadow_evaluation" / "gate_result.json", default={}) or {}
        if str(gate_result.get("overall_status")) == "pass":
            passing.append(report_dir)
    return passing[-1] if passing else None


def _load_metric_registry(repo_root: Path) -> dict[str, Any]:
    benchmark_cfg = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    metric_registry_path = resolve_repo_path(repo_root, benchmark_cfg["metric_registry"])
    return load_yaml(metric_registry_path)


def _stability_envelope_value(summary: Any, trend_preference: str | None) -> Any:
    if not isinstance(summary, dict):
        return summary
    if "min" not in summary or "max" not in summary:
        return summary.get("mean")
    if trend_preference == "lower_better":
        return summary["max"]
    if trend_preference == "higher_better":
        return summary["min"]
    return summary.get("mean")


def _build_stability_envelope_baseline(repo_root: Path) -> dict[str, Any] | None:
    stability_report_path = repo_root / "benchmark" / "stage6_online_shadow_stability_report.json"
    stability_report = load_json(stability_report_path, default={}) or {}
    if str(stability_report.get("readiness")) != "ready":
        return None

    metric_registry = _load_metric_registry(repo_root)
    metric_meta = (metric_registry.get("metrics") or {}) if isinstance(metric_registry, dict) else {}
    tracked_metrics = list(
        dict.fromkeys(
            list(stability_report.get("hard_gate_metrics") or [])
            + list(stability_report.get("soft_guardrail_metrics") or [])
        )
    )
    case_rows = list(stability_report.get("cases") or [])
    if not case_rows or not tracked_metrics:
        return None

    cases_payload: dict[str, Any] = {}
    for case_row in case_rows:
        case_id = str(case_row.get("case_id") or "").strip()
        if not case_id:
            continue
        metrics_payload: dict[str, Any] = {}
        for metric_name in tracked_metrics:
            trend_preference = (metric_meta.get(metric_name) or {}).get("trend_preference")
            value = _stability_envelope_value(case_row.get(metric_name), trend_preference)
            if value is not None:
                metrics_payload[metric_name] = value
        if metrics_payload:
            cases_payload[case_id] = {
                "status": "pass",
                "case_status": "ready",
                "metrics": metrics_payload,
            }

    if not cases_payload:
        return None

    return {
        "benchmark_version": "system_benchmark_v1",
        "generated_at_utc": _utc_now_iso(),
        "set": "stage6_online_shadow_golden",
        "baseline_source": "stage6_online_shadow_stability_envelope",
        "baseline_source_report": str(stability_report_path),
        "cases": cases_payload,
    }


def ensure_stage6_online_shadow_baseline(
    repo_root: str | Path,
    *,
    repin: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    baseline_metrics_path = repo_root / BASELINE_METRICS
    baseline_commit_path = repo_root / BASELINE_COMMIT
    decision_path = repo_root / BASELINE_DECISION
    readiness = load_json(repo_root / "benchmark" / "stage6_online_shadow_stability_result.json", default={}) or {}
    if str(readiness.get("overall_status")) != "ready":
        raise RuntimeError("Stage 6 online shadow golden baseline is blocked because stage6_online_shadow_stability_result.json is not ready.")

    selected_run = _latest_pass_shadow_smoke_run(repo_root)
    decision = {
        "schema_version": "stage6_online_shadow_golden_baseline_decision_v1",
        "generated_at_utc": _utc_now_iso(),
        "readiness_source": str(repo_root / "benchmark" / "stage6_online_shadow_stability_result.json"),
        "readiness_status": readiness.get("overall_status"),
        "selected_run": str(selected_run) if selected_run is not None else None,
        "repin_requested": bool(repin),
        "repin_status": "deferred",
        "reason": None,
        "baseline_metrics_path": str(baseline_metrics_path),
        "baseline_commit_path": str(baseline_commit_path),
    }

    if selected_run is None:
        decision["reason"] = "no passing stage6_online_shadow_smoke report found"
        dump_json(decision_path, decision)
        return decision

    selected_snapshot = selected_run / "online_e2e" / "stage6_online_shadow_evaluation" / "baseline_snapshot.json"
    if not selected_snapshot.exists():
        decision["reason"] = f"baseline snapshot missing: {selected_snapshot}"
        dump_json(decision_path, decision)
        return decision

    if repin or not baseline_metrics_path.exists():
        baseline_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        envelope_payload = _build_stability_envelope_baseline(repo_root)
        if envelope_payload is not None:
            dump_json(baseline_metrics_path, envelope_payload)
            baseline_source = "stability_envelope"
        else:
            shutil.copy2(selected_snapshot, baseline_metrics_path)
            baseline_source = "latest_passing_smoke_snapshot"
        commit_payload = {
            "schema_version": "stage6_online_shadow_baseline_commit_v1",
            "generated_at_utc": _utc_now_iso(),
            "benchmark_version": "system_benchmark_v1_stage6_online_shadow",
            "source_report_dir": str(selected_run),
            "source_snapshot": str(selected_snapshot),
            "readiness_source": str(repo_root / "benchmark" / "stage6_online_shadow_stability_report.json"),
            "baseline_source": baseline_source,
            "selected_cases": list(SHADOW_CASES),
            "note": "Pinned from the ready online-shadow stability envelope when available; otherwise falls back to the latest passing online shadow smoke snapshot.",
        }
        dump_json(baseline_commit_path, commit_payload)
        decision["repin_status"] = "approved"
        decision["reason"] = (
            "baseline pinned from stability envelope derived from the ready online shadow repeatability campaign"
            if baseline_source == "stability_envelope"
            else "baseline pinned from latest passing online shadow smoke report"
        )
    else:
        decision["repin_status"] = "unchanged"
        decision["reason"] = "baseline already present; reuse existing pin"

    dump_json(decision_path, decision)
    return decision


def _materialize_golden_eval_config(online_report_root: Path) -> Path:
    runtime_cfg_path = online_report_root / "stage6_online_shadow_runtime_config.yaml"
    runtime_cfg = load_yaml(runtime_cfg_path)
    sets = dict(runtime_cfg.get("sets") or {})
    smoke_set_path = Path(str(sets["stage6_online_shadow_smoke"]))
    smoke_payload = load_yaml(smoke_set_path)
    golden_set_path = smoke_set_path.parent / "stage6_online_shadow_golden.yaml"
    _write_yaml(
        golden_set_path,
        {
            "set_name": "stage6_online_shadow_golden",
            "description": "Pinned online shadow golden subset on the promoted two-case rollout.",
            "cases": list(smoke_payload.get("cases") or SHADOW_CASES),
        },
    )
    runtime_cfg["sets"] = sets
    runtime_cfg["sets"]["stage6_online_shadow_golden"] = str(golden_set_path)
    golden_cfg_path = online_report_root / "stage6_online_shadow_golden_runtime_config.yaml"
    _write_yaml(golden_cfg_path, runtime_cfg)
    return golden_cfg_path


def _build_stage6_online_shadow_golden_report(
    *,
    repo_root: Path,
    smoke_result: dict[str, Any],
    golden_eval: dict[str, Any],
    baseline_decision: dict[str, Any],
) -> dict[str, Any]:
    online_report_root = Path(str((smoke_result.get("gate_result") or {}).get("online_report_dir")))
    smoke_report = smoke_result.get("shadow_report") or {}
    gate_result = golden_eval.get("gate_result") or {}
    report = {
        "schema_version": "stage6_online_shadow_golden_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "stage": "6_online_shadow_golden",
        "description": "Pinned online shadow golden regression gate on the promoted two-case subset.",
        "subset_cases": list(SHADOW_CASES),
        "baseline_decision_path": str(repo_root / BASELINE_DECISION),
        "baseline_metrics_path": str(repo_root / BASELINE_METRICS),
        "baseline_commit_path": str(repo_root / BASELINE_COMMIT),
        "selected_baseline_run": baseline_decision.get("selected_run"),
        "online_report_dir": str(online_report_root),
        "smoke_result_path": str(repo_root / "benchmark" / "stage6_online_shadow_smoke_result.json"),
        "golden_evaluation_report_dir": str(golden_eval.get("report_dir")),
        "overall_status": gate_result.get("overall_status"),
        "cases": list(smoke_report.get("cases") or []),
        "notes": [
            "Baseline remains the only execution authority; MPC shadow stays proposal-only.",
            "Golden gate compares the current online shadow subset against the pinned online shadow baseline snapshot.",
        ],
    }
    dump_json(repo_root / "benchmark" / "stage6_online_shadow_golden_report.json", report)
    return report


def run_stage6_online_shadow_golden_gate(
    repo_root: str | Path,
    *,
    repin_baseline: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    baseline_decision = ensure_stage6_online_shadow_baseline(repo_root, repin=repin_baseline)
    smoke_result = run_stage6_online_shadow_smoke(repo_root)
    online_report_root = Path(str((smoke_result.get("gate_result") or {}).get("online_report_dir")))
    golden_cfg = _materialize_golden_eval_config(online_report_root)

    golden_eval = run_benchmark(
        repo_root=repo_root,
        config_path=golden_cfg,
        set_name="stage6_online_shadow_golden",
        compare_baseline_path=(repo_root / BASELINE_METRICS),
        report_dir=online_report_root / "stage6_online_shadow_golden_evaluation",
    )
    golden_report = _build_stage6_online_shadow_golden_report(
        repo_root=repo_root,
        smoke_result=smoke_result,
        golden_eval=golden_eval,
        baseline_decision=baseline_decision,
    )
    gate = golden_eval.get("gate_result") or {}
    mode_resolution = load_json(online_report_root / "mode_resolution.json", default=[]) or []
    runtime_failed = [
        str(row.get("scenario_id"))
        for row in mode_resolution
        if str(row.get("status")) != "success"
    ]
    failed_cases = sorted(set((gate.get("failed_cases") or []) + runtime_failed))
    warning_cases = sorted(set(gate.get("warning_cases") or []))
    overall_status = "pass"
    if failed_cases:
        overall_status = "fail"
    elif warning_cases:
        overall_status = "partial"

    payload = {
        "schema_version": "stage6_online_shadow_golden_result_v1",
        "generated_at_utc": _utc_now_iso(),
        "stage": "6_online_shadow_golden",
        "benchmark_mode": "online_e2e",
        "overall_status": overall_status,
        "failed_cases": failed_cases,
        "warning_cases": warning_cases,
        "subset_cases": list(SHADOW_CASES),
        "baseline_decision_path": str(repo_root / BASELINE_DECISION),
        "baseline_metrics_path": str(repo_root / BASELINE_METRICS),
        "baseline_commit_path": str(repo_root / BASELINE_COMMIT),
        "online_report_dir": str(online_report_root),
        "golden_evaluation_report_dir": str(golden_eval.get("report_dir")),
        "report_path": str(repo_root / "benchmark" / "stage6_online_shadow_golden_report.json"),
    }
    dump_json(repo_root / "benchmark" / "stage6_online_shadow_golden_result.json", payload)
    return {
        "baseline_decision": baseline_decision,
        "smoke_result": smoke_result,
        "golden_evaluation": golden_eval,
        "golden_report": golden_report,
        "gate_result": payload,
    }
