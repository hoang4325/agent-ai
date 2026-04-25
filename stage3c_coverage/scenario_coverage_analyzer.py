from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .evaluation_stage3c_coverage import build_stage3c_run_metrics, load_json, select_best_run


def _safe_load_json(path: str | Path) -> Dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return load_json(path)


def analyze_stage3c_coverage(
    *,
    stage3c_root: str | Path,
    stage3b_junction_dir: str | Path | None = None,
    stage3a_junction_dir: str | Path | None = None,
) -> Dict[str, Any]:
    stage3c_root = Path(stage3c_root)
    run_metrics: List[Dict[str, Any]] = []
    if stage3c_root.exists():
        for run_dir in sorted(path for path in stage3c_root.iterdir() if path.is_dir()):
            if not (run_dir / "execution_timeline.jsonl").exists():
                continue
            run_metrics.append(build_stage3c_run_metrics(run_dir))

    right_runs = [
        metrics
        for metrics in run_metrics
        if int(metrics["requested_behavior_counts"].get("commit_lane_change_right", 0)) > 0
    ]
    left_runs = [
        metrics
        for metrics in run_metrics
        if int(metrics["requested_behavior_counts"].get("commit_lane_change_left", 0)) > 0
    ]
    junction_runs = [metrics for metrics in run_metrics if int(metrics.get("junction_frames", 0)) > 0]

    best_right_run = select_best_run(right_runs, direction="right", require_success=True)
    best_left_run = select_best_run(left_runs, direction="left", require_success=True)
    best_stop_run = max(
        run_metrics,
        key=lambda item: (
            int(item.get("stop_successes", 0)),
            -int(item.get("stop_overshoots", 0)),
            int(item.get("lane_change_successes", 0)),
            int(item.get("num_frames", 0)),
            str(item.get("run_name", "")),
        ),
        default=None,
    )
    best_junction_run = max(
        junction_runs,
        key=lambda item: (int(item.get("num_frames", 0)), int(item.get("stop_successes", 0)), str(item.get("run_name", ""))),
        default=None,
    )

    stage3b_junction_summary = None
    if stage3b_junction_dir is not None:
        stage3b_junction_summary = _safe_load_json(Path(stage3b_junction_dir) / "evaluation_summary.json")
    stage3a_junction_summary = None
    if stage3a_junction_dir is not None:
        stage3a_junction_summary = _safe_load_json(Path(stage3a_junction_dir) / "evaluation_summary.json")

    gap_analysis = {
        "right_positive": {
            "status": "pass" if best_right_run is not None else "missing_verified_success",
            "best_lane_change_run": None if best_right_run is None else best_right_run["run_name"],
            "observed_runs": [metrics["run_name"] for metrics in right_runs],
            "note": (
                "Right-positive closed-loop baseline exists."
                if best_right_run is not None
                else "No verified right-positive lane-change success found."
            ),
        },
        "left_positive": {
            "status": "pass" if best_left_run is not None else "missing_scenario_or_execution",
            "best_lane_change_run": None if best_left_run is None else best_left_run["run_name"],
            "observed_runs": [metrics["run_name"] for metrics in left_runs],
            "note": (
                "Left-positive closed-loop verified."
                if best_left_run is not None
                else "No left-positive closed-loop evidence yet."
            ),
        },
        "junction_execution": {
            "status": "pass" if best_junction_run is not None else "missing_closed_loop_execution",
            "best_run": None if best_junction_run is None else best_junction_run["run_name"],
            "stage3b_ready": bool(stage3b_junction_summary and int(stage3b_junction_summary.get("junction_frames", 0)) > 0),
            "stage3a_ready": bool(stage3a_junction_summary and int(stage3a_junction_summary.get("junction_frames", 0)) > 0),
            "note": (
                "Junction closed-loop evidence exists."
                if best_junction_run is not None
                else "Junction behavior is replay-ready in Stage 3A/3B but lacks Stage 3C closed-loop evidence."
            ),
        },
        "stop_controller": {
            "status": (
                "pass"
                if best_stop_run is not None
                and int(best_stop_run.get("stop_successes", 0)) > 0
                and int(best_stop_run.get("stop_overshoots", 0)) == 0
                else "partial"
            ),
            "best_stop_run": None if best_stop_run is None else best_stop_run["run_name"],
            "note": (
                "At least one run achieved stop success without derived overshoot."
                if best_stop_run is not None and int(best_stop_run.get("stop_successes", 0)) > 0
                else "Stop controller needs more verified evidence."
            ),
        },
        "lane_change_completion": {
            "status": "pass" if best_right_run is not None else "partial",
            "best_reference_run": None if best_right_run is None else best_right_run["run_name"],
            "note": (
                "Completion now tracks target-lane stability plus lateral-shift thresholds."
                if best_right_run is not None
                else "Completion refinement is implemented but lacks a fresh verified run under the new criteria."
            ),
        },
    }

    execution_coverage_summary = {
        "right_positive": None
        if best_right_run is None
        else {
            "run_name": best_right_run["run_name"],
            "lane_change_attempts": best_right_run["lane_change_attempts"],
            "lane_change_successes": best_right_run["lane_change_successes"],
            "lane_change_failures": best_right_run["lane_change_failures"],
            "stop_successes": best_right_run["stop_successes"],
            "mean_lane_change_time_s": best_right_run["mean_lane_change_time_s"],
        },
        "left_positive": None
        if best_left_run is None
        else {
            "run_name": best_left_run["run_name"],
            "lane_change_attempts": best_left_run["lane_change_attempts"],
            "lane_change_successes": best_left_run["lane_change_successes"],
            "lane_change_failures": best_left_run["lane_change_failures"],
        },
        "junction_execution": None
        if best_junction_run is None
        else {
            "run_name": best_junction_run["run_name"],
            "num_frames": best_junction_run["num_frames"],
            "stop_successes": best_junction_run["stop_successes"],
            "junction_frames": best_junction_run["junction_frames"],
        },
        "stop_controller_metrics": None
        if best_stop_run is None
        else {
            "run_name": best_stop_run["run_name"],
            "stop_attempts": best_stop_run["stop_attempts"],
            "stop_successes": best_stop_run["stop_successes"],
            "stop_overshoots": best_stop_run["stop_overshoots"],
            "stop_aborts": best_stop_run["stop_aborts"],
        },
        "lane_change_completion_metrics": None
        if best_right_run is None
        else {
            "run_name": best_right_run["run_name"],
            "mean_lane_change_time_s": best_right_run["mean_lane_change_time_s"],
            "max_lane_change_lateral_shift_m": best_right_run["max_lane_change_lateral_shift_m"],
            "lane_change_successes": best_right_run["lane_change_successes"],
        },
    }

    return {
        "stage3c_root": str(stage3c_root),
        "run_inventory": run_metrics,
        "gap_analysis": gap_analysis,
        "execution_coverage_summary": execution_coverage_summary,
        "stage3b_junction_summary": stage3b_junction_summary,
        "stage3a_junction_summary": stage3a_junction_summary,
    }
