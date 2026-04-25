from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGET_METRICS = [
    "stale_tick_rate",
    "mean_total_loop_latency_ms",
    "dropped_frames",
    "state_reset_count",
    "stop_success_rate",
    "lane_change_success_rate",
    "behavior_execution_mismatch_rate",
    "route_context_consistency",
    "critical_actor_detected_rate",
]


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def classify_metric(metric: str, num_success_samples: int, has_values: bool) -> tuple[str, str, str]:
    if num_success_samples == 0 or not has_values:
        return (
            "unstable",
            "diagnostic_only",
            "No successful online samples with valid values yet.",
        )

    if metric in {"state_reset_count", "route_context_consistency", "critical_actor_detected_rate"}:
        return (
            "acceptable",
            "soft_guardrail",
            "Potentially gateable later, but online sample count is still limited.",
        )

    if metric in {"stale_tick_rate", "mean_total_loop_latency_ms", "dropped_frames"}:
        return (
            "flaky",
            "soft_guardrail",
            "Runtime-sensitive metric; keep as soft guardrail until environment variance is controlled.",
        )

    return (
        "flaky",
        "diagnostic_only",
        "Behavior metric not yet proven repeatable in online mode.",
    )


def run_registry(stability_report_path: Path, output_path: Path) -> dict[str, Any]:
    stability = _read_json(stability_report_path) or {}
    cases = stability.get("cases") or []

    num_success_samples = 0
    for case in cases:
        for run_result in case.get("run_results") or []:
            if run_result.get("stage4_status") == "success":
                num_success_samples += 1

    registry: dict[str, Any] = {}
    for metric in TARGET_METRICS:
        has_values = False
        for case in cases:
            for run_result in case.get("run_results") or []:
                if metric in run_result and run_result.get(metric) is not None:
                    has_values = True
                    break
            if has_values:
                break

        stability_class, gate_reco, rationale = classify_metric(metric, num_success_samples, has_values)
        registry[metric] = {
            "stability_class": stability_class,
            "gate_recommendation": gate_reco,
            "rationale": rationale,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Build online metric stability registry from stability reports.")
    parser.add_argument("--stability-report", default="benchmark/reports/online_stability/online_smoke_stability_report.json")
    parser.add_argument("--output", default="benchmark/reports/online_stability/online_metric_stability_registry.json")
    args = parser.parse_args()

    payload = run_registry(Path(args.stability_report), Path(args.output))
    print(json.dumps({"metrics": len(payload)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
