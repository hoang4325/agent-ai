from __future__ import annotations

import argparse
import fnmatch
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Stage10 nominal driving runs into Table 1 metrics."
    )
    parser.add_argument(
        "--report-root",
        default=str(Path("benchmark") / "reports"),
        help="Root directory containing per-run stage10_driving_metrics.json files",
    )
    parser.add_argument(
        "--run-glob",
        default="*",
        help="Shell-style glob matched against run directory names",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write the computed Table 1 summary as JSON",
    )
    return parser.parse_args()


def _mean_std(values: List[float]) -> Dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None}
    if len(values) == 1:
        return {"count": 1, "mean": round(values[0], 6), "std": 0.0}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 6),
        "std": round(statistics.stdev(values), 6),
    }


def _load_runs(report_root: Path, run_glob: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for metrics_path in sorted(report_root.glob("*/stage10_driving_metrics.json")):
        run_dir = metrics_path.parent
        if not fnmatch.fnmatch(run_dir.name, run_glob):
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        runtime = payload.get("runtime") or {}
        route = payload.get("route") or {}
        runs.append(
            {
                "run_name": run_dir.name,
                "map": payload.get("map"),
                "agent_mode": payload.get("agent_mode"),
                "frames": int(payload.get("frames") or 0),
                "route_mode": route.get("route_mode"),
                "route_completion_pct": float(payload.get("route_completion_pct") or 0.0),
                "collision_count": float(payload.get("collision_count") or 0.0),
                "collision_rate_per_km": (
                    float(payload["collision_rate_per_km"])
                    if payload.get("collision_rate_per_km") is not None
                    else None
                ),
                "scenario_success_rate": float(payload.get("scenario_success_rate") or 0.0),
                "avg_bev_inference_ms": float(runtime.get("avg_bev_inference_ms") or 0.0),
                "avg_detections_per_frame": float(runtime.get("avg_detections_per_frame") or 0.0),
            }
        )
    return runs


def main() -> int:
    args = _parse_args()
    report_root = Path(args.report_root)
    runs = _load_runs(report_root, str(args.run_glob))
    if not runs:
        raise SystemExit(
            f"No nominal runs found under {report_root} matching run_glob={args.run_glob!r}"
        )

    summary = {
        "schema_version": "stage10_nominal_table1_summary_v1",
        "report_root": str(report_root),
        "run_glob": str(args.run_glob),
        "num_runs": len(runs),
        "maps": sorted({str(run.get('map') or '') for run in runs}),
        "route_modes": sorted({str(run.get('route_mode') or '') for run in runs}),
        "frames_per_run": sorted({int(run.get('frames') or 0) for run in runs}),
        "metrics": {
            "route_completion_pct": _mean_std([run["route_completion_pct"] for run in runs]),
            "collision_count": _mean_std([run["collision_count"] for run in runs]),
            "collision_rate_per_km": _mean_std(
                [run["collision_rate_per_km"] for run in runs if run["collision_rate_per_km"] is not None]
            ),
            "scenario_success_rate": _mean_std([run["scenario_success_rate"] for run in runs]),
            "avg_bev_inference_ms": _mean_std([run["avg_bev_inference_ms"] for run in runs]),
            "avg_detections_per_frame": _mean_std([run["avg_detections_per_frame"] for run in runs]),
        },
        "runs": runs,
    }

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== TABLE 1 RUNS ===")
    print("run,frames,route_mode,rc_pct,collision_count,collision_rate_per_km,success_rate,avg_bev_ms,avg_det_per_frame")
    for run in runs:
        collision_rate_text = (
            ""
            if run["collision_rate_per_km"] is None
            else f"{run['collision_rate_per_km']:.6f}"
        )
        print(
            f"{run['run_name']},{run['frames']},{run['route_mode']},"
            f"{run['route_completion_pct']:.3f},{run['collision_count']:.3f},"
            f"{collision_rate_text},"
            f"{run['scenario_success_rate']:.3f},{run['avg_bev_inference_ms']:.3f},"
            f"{run['avg_detections_per_frame']:.3f}"
        )

    print("\n=== TABLE 1 SUMMARY ===")
    print(json.dumps(summary["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
