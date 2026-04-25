from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.runner_core import run_benchmark  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Agent-AI system benchmark scaffold.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "benchmark" / "benchmark_v1.yaml"),
        help="Path to benchmark config YAML.",
    )
    parser.add_argument(
        "--set",
        dest="set_name",
        required=True,
        choices=["smoke", "golden", "stress", "failure_replay"],
        help="Benchmark set to run.",
    )
    parser.add_argument(
        "--compare-baseline",
        default=None,
        help="Optional baseline metrics JSON path.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Optional explicit report output directory.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    result = run_benchmark(
        repo_root=REPO_ROOT,
        config_path=args.config,
        set_name=args.set_name,
        compare_baseline_path=args.compare_baseline,
        report_dir=args.report_dir,
    )

    gate_result = result["gate_result"]
    console_summary = {
        "benchmark_version": gate_result["benchmark_version"],
        "set": gate_result["set"],
        "overall_status": gate_result["overall_status"],
        "failed_cases": gate_result["failed_cases"],
        "warning_cases": gate_result["warning_cases"],
        "report_dir": result["report_dir"],
    }
    print(json.dumps(console_summary, indent=2))
    return 0 if gate_result["overall_status"] in {"pass", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
