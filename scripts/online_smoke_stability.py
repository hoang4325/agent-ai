from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


@dataclass
class RunCaseStatus:
    run_dir: str
    case_id: str
    stage4_status: str
    has_online_summary: bool
    stale_tick_rate: float | None
    mean_total_loop_latency_ms: float | None
    state_reset_count: int | None


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _latest_case_status_by_run(run_dir: Path) -> dict[str, str]:
    trace = _read_jsonl(run_dir / "benchmark_execution_trace.jsonl")
    status: dict[str, str] = {}
    for row in trace:
        if row.get("step") != "stage4":
            continue
        status[str(row.get("case_id"))] = str(row.get("status"))
    return status


def _collect_runs(reports_root: Path) -> list[Path]:
    return sorted([p for p in reports_root.glob("e2e_online_e2e_smoke_*") if p.is_dir()])


def _classification(success_rate: float) -> str:
    if success_rate >= 0.95:
        return "stable"
    if success_rate >= 0.75:
        return "acceptable"
    if success_rate >= 0.4:
        return "flaky"
    return "unstable"


def _variance(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def run_analysis(reports_root: Path, output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    runs = _collect_runs(reports_root)
    by_case: dict[str, list[RunCaseStatus]] = {}

    for run in runs:
        stage4_status = _latest_case_status_by_run(run)
        for case_id, status in stage4_status.items():
            online_summary = _read_json(run / "case_runs" / case_id / "stage4" / "online_run_summary.json") or {}
            record = RunCaseStatus(
                run_dir=run.name,
                case_id=case_id,
                stage4_status=status,
                has_online_summary=bool(online_summary),
                stale_tick_rate=(None if online_summary.get("stale_tick_rate") is None else float(online_summary.get("stale_tick_rate"))),
                mean_total_loop_latency_ms=(None if online_summary.get("mean_total_loop_latency_ms") is None else float(online_summary.get("mean_total_loop_latency_ms"))),
                state_reset_count=(None if online_summary.get("state_resets") is None else int(online_summary.get("state_resets"))),
            )
            by_case.setdefault(case_id, []).append(record)

    stability_cases: list[dict[str, Any]] = []
    variance_cases: list[dict[str, Any]] = []

    for case_id, rows in sorted(by_case.items()):
        total = len(rows)
        success = sum(1 for r in rows if r.stage4_status == "success")
        success_rate = (float(success) / float(total)) if total else 0.0

        stale_vals = [r.stale_tick_rate for r in rows if r.stale_tick_rate is not None]
        latency_vals = [r.mean_total_loop_latency_ms for r in rows if r.mean_total_loop_latency_ms is not None]
        reset_vals = [float(r.state_reset_count) for r in rows if r.state_reset_count is not None]

        run_results = [
            {
                "run_dir": r.run_dir,
                "stage4_status": r.stage4_status,
                "has_online_summary": r.has_online_summary,
                "stale_tick_rate": r.stale_tick_rate,
                "mean_total_loop_latency_ms": r.mean_total_loop_latency_ms,
                "state_reset_count": r.state_reset_count,
            }
            for r in rows
        ]

        stability = _classification(success_rate)
        mean_metric_variance = mean([
            _variance(stale_vals)["std"] or 0.0,
            _variance(latency_vals)["std"] or 0.0,
            _variance(reset_vals)["std"] or 0.0,
        ])

        stability_cases.append(
            {
                "case_id": case_id,
                "run_results": run_results,
                "stability_class": stability,
                "artifact_completeness_rate": float(sum(1 for r in rows if r.has_online_summary) / float(total)) if total else 0.0,
                "mean_metric_variance": float(mean_metric_variance),
                "notes": ["stage4_import_or_runtime_failures_present"] if stability in {"flaky", "unstable"} else [],
            }
        )

        variance_cases.append(
            {
                "case_id": case_id,
                "metrics": {
                    "stale_tick_rate": _variance([float(v) for v in stale_vals]),
                    "mean_total_loop_latency_ms": _variance([float(v) for v in latency_vals]),
                    "state_reset_count": _variance([float(v) for v in reset_vals]),
                },
                "stability_class": stability,
                "recommendation": (
                    "keep_online_smoke_only" if stability in {"flaky", "unstable"} else "candidate_online_golden"
                ),
            }
        )

    stability_report = {
        "benchmark_version": "system_benchmark_v1",
        "num_runs": len(runs),
        "runs": [r.name for r in runs],
        "cases": stability_cases,
    }

    variance_report = {
        "benchmark_version": "system_benchmark_v1",
        "cases": variance_cases,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "online_smoke_stability_report.json").write_text(json.dumps(stability_report, indent=2) + "\n", encoding="utf-8")
    (output_root / "online_case_variance_report.json").write_text(json.dumps(variance_report, indent=2) + "\n", encoding="utf-8")
    return stability_report, variance_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze online_e2e_smoke run-to-run stability.")
    parser.add_argument("--reports-root", default="benchmark/reports")
    parser.add_argument("--output-root", default="benchmark/reports/online_stability")
    args = parser.parse_args()

    stability, _ = run_analysis(Path(args.reports_root), Path(args.output_root))
    print(json.dumps({"num_runs": stability.get("num_runs"), "num_cases": len(stability.get("cases", []))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
