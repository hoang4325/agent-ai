from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_json_from_stdout(stdout: str) -> Dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Cannot locate JSON payload in benchmark runner stdout.")
    return json.loads(stdout[start : end + 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage4 online e2e revalidation and summarize worker/stale status.")
    parser.add_argument("--config", default="benchmark/benchmark_v1.yaml")
    parser.add_argument("--set", default="online_e2e_smoke")
    parser.add_argument("--benchmark-mode", default="online_e2e")
    parser.add_argument("--output", default="outputs/stage4_infra_smoke/stage4_online_revalidation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "run_system_benchmark_e2e.py").resolve()),
        "--config",
        str(args.config),
        "--set",
        str(args.set),
        "--benchmark-mode",
        str(args.benchmark_mode),
        "--execute-stages",
    ]

    proc = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    runner_payload = _extract_json_from_stdout(proc.stdout)
    report_root = Path(runner_payload["report_dir"]).resolve().parents[0]
    if report_root.name == "evaluation":
        report_root = report_root.parent

    benchmark_report = _read_json(report_root / "evaluation" / "benchmark_report.json")
    case_runs_dir = report_root / "case_runs"

    worker_exit_codes: List[int | None] = []
    behavior_updates = 0
    execution_updates = 0
    stale_rates: List[float] = []
    remaining_blockers: List[str] = []

    for case in benchmark_report.get("cases", []):
        case_id = str(case["scenario_id"])
        stage4_summary_path = case_runs_dir / case_id / "stage4" / "online_run_summary.json"
        if not stage4_summary_path.exists():
            remaining_blockers.append(f"missing_stage4_summary:{case_id}")
            continue
        stage4_summary = _read_json(stage4_summary_path)
        behavior_updates += int(stage4_summary.get("num_behavior_updates", 0))
        execution_updates += int(stage4_summary.get("num_ticks", 0))
        worker_exit_codes.append(stage4_summary.get("stage1_summary", {}).get("worker_exit_code"))
        if int(stage4_summary.get("stage1_summary", {}).get("inferred_samples", 0)) <= 0:
            remaining_blockers.append(f"no_stage1_inference:{case_id}")

        stale_metric = case.get("metrics", {}).get("stale_tick_rate", {}).get("value")
        if stale_metric is not None:
            stale_rates.append(float(stale_metric))

    stale_tick_rate = (sum(stale_rates) / len(stale_rates)) if stale_rates else None
    worker_stable = bool(
        worker_exit_codes
        and all(code in (0, None) for code in worker_exit_codes)
        and not any(item.startswith("no_stage1_inference:") for item in remaining_blockers)
    )

    if stale_tick_rate is not None and stale_tick_rate > 0.2:
        remaining_blockers.append(f"stale_tick_rate_high:{stale_tick_rate}")

    payload = {
        "online_run_status": runner_payload.get("overall_status", "unknown"),
        "worker_stable": worker_stable,
        "stale_tick_rate": stale_tick_rate,
        "behavior_updates": behavior_updates,
        "execution_updates": execution_updates,
        "remaining_blockers": sorted(set(remaining_blockers)),
        "report_dir": str(report_root),
        "runner_exit_code": int(proc.returncode),
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        "[stage4-online-revalidation] "
        f"status={payload['online_run_status']} "
        f"worker_stable={payload['worker_stable']} "
        f"stale_tick_rate={payload['stale_tick_rate']}"
    )
    if payload["online_run_status"] == "pass" and payload["worker_stable"] and not payload["remaining_blockers"]:
        return
    raise SystemExit(2)


if __name__ == "__main__":
    main()
