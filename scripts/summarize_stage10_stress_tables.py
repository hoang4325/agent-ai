from __future__ import annotations

import argparse
import fnmatch
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List


RUN_SUFFIX_RE = re.compile(r"(?:_seed\d+)?_run\d+$")

STATISTIC_FIELDS = (
    "route_progress_m",
    "route_completion_rate",
    "distance_traveled_m",
    "collision_count",
    "collision_episode_count",
    "collision_sensor_event_count",
    "lane_invasion_count",
    "legal_lane_crossing_count",
    "illegal_lane_invasion_count",
    "maneuver_illegal_lane_invasion_count",
    "post_maneuver_illegal_lane_invasion_count",
    "unknown_lane_crossing_count",
    "offroad_rate",
    "mean_abs_longitudinal_jerk_mps3",
    "max_abs_longitudinal_jerk_mps3",
    "episode_duration_s",
    "maneuver_duration_s",
    "agreement_rate",
    "agent_fallback_rate",
    "agent_timeout_rate",
    "assist_query_rejection_rate",
    "safety_arbitration_rejection_rate",
    "end_to_end_query_success_rate",
    "assist_p95_api_call_ms",
    "assist_over_step_budget_rate",
    "stale_response_discard_rate",
    "control_loop_p95_ms",
    "control_loop_over_budget_rate",
    "lane_change_completion_time_s",
    "lane_change_completed",
)

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Stage10 stress runs into Table 2 and Table 3."
    )
    parser.add_argument(
        "--report-root",
        required=True,
        help="Root directory containing per-case stress run folders",
    )
    parser.add_argument(
        "--run-glob",
        default="*",
        help="Shell-style glob matched against run directory names",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write the computed stress summary as JSON",
    )
    return parser.parse_args()


def _case_name(run_name: str) -> str:
    return RUN_SUFFIX_RE.sub("", run_name)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def _load_rows(report_root: Path, run_glob: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for driving_path in sorted(report_root.glob("*/stage10_driving_metrics.json")):
        run_dir = driving_path.parent
        if not fnmatch.fnmatch(run_dir.name, run_glob):
            continue
        evaluation_path = run_dir / "stage10_agent_live_evaluation.json"
        assist_path = run_dir / "stage10_agent_assist_evaluation.json"

        driving = json.loads(driving_path.read_text(encoding="utf-8"))
        evaluation = (
            json.loads(evaluation_path.read_text(encoding="utf-8"))
            if evaluation_path.exists()
            else {}
        )
        assist = json.loads(assist_path.read_text(encoding="utf-8")) if assist_path.exists() else {}
        low_ttc_analysis = evaluation.get("low_ttc_analysis") or {}
        comfort = driving.get("comfort") or {}
        assist_latency = assist.get("latency") or {}
        compare_latency = evaluation.get("latency") or {}
        control_loop_latency = (
            assist.get("control_loop_latency")
            or compare_latency.get("control_loop")
            or ((driving.get("runtime") or {}).get("control_loop_latency"))
            or {}
        )
        lane_change_maneuver = assist.get("lane_change_maneuver") or {}
        queried_frames = int(
            evaluation.get("agent_queried_frames")
            or assist.get("agent_query_frames")
            or 0
        )
        sim_frames = int(
            evaluation.get("sim_frames")
            or assist.get("sim_frames")
            or driving.get("frames")
            or 0
        )
        query_ratio = queried_frames / sim_frames if sim_frames else 0.0
        collision_count = int(driving.get("collision_count") or 0)
        collision_episode_count = int(
            driving.get("collision_episode_count")
            if driving.get("collision_episode_count") is not None
            else collision_count
        )
        safety_outcome = "Collision" if collision_episode_count > 0 else "Collision-free"
        route = driving.get("route") or {}

        rows.append(
            {
                "run_name": run_dir.name,
                "case": _case_name(run_dir.name),
                "random_seed": driving.get("random_seed", evaluation.get("random_seed", assist.get("random_seed"))),
                "frames": sim_frames or int(driving.get("frames") or 0),
                "route_progress_m": _optional_float(route.get("route_progress_m")),
                "route_completion_rate": _optional_float(driving.get("route_completion_rate")),
                "distance_traveled_m": _optional_float(route.get("distance_traveled_m")),
                "collision_count": collision_count,
                "collision_episode_count": collision_episode_count,
                "collision_sensor_event_count": int(
                    driving.get("collision_sensor_event_count")
                    if driving.get("collision_sensor_event_count") is not None
                    else collision_count
                ),
                "lane_invasion_count": int(driving.get("lane_invasion_count") or 0),
                "legal_lane_crossing_count": int(
                    driving.get("legal_lane_crossing_count") or 0
                ),
                "illegal_lane_invasion_count": int(
                    driving.get("illegal_lane_invasion_count") or 0
                ),
                "maneuver_illegal_lane_invasion_count": int(
                    driving.get("maneuver_illegal_lane_invasion_count") or 0
                ),
                "post_maneuver_illegal_lane_invasion_count": int(
                    driving.get("post_maneuver_illegal_lane_invasion_count") or 0
                ),
                "unknown_lane_crossing_count": int(
                    driving.get("unknown_lane_crossing_count") or 0
                ),
                "offroad_rate": _optional_float(driving.get("offroad_rate")),
                "mean_abs_longitudinal_jerk_mps3": _optional_float(
                    comfort.get("mean_abs_longitudinal_jerk_mps3")
                ),
                "max_abs_longitudinal_jerk_mps3": _optional_float(
                    comfort.get("max_abs_longitudinal_jerk_mps3")
                ),
                "episode_duration_s": _optional_float(
                    driving.get("episode_duration_s")
                    if driving.get("episode_duration_s") is not None
                    else driving.get("maneuver_duration_s")
                ),
                "maneuver_duration_s": _optional_float(driving.get("maneuver_duration_s")),
                "low_ttc_frames": int(low_ttc_analysis.get("total_low_ttc_frames") or 0),
                "safety_outcome": safety_outcome,
                "agreement_rate": _optional_float(evaluation.get("agreement_rate")),
                "disagreement_rate": _optional_float(evaluation.get("disagreement_rate")),
                "useful_disagreement_count": (
                    int(evaluation["useful_disagreement_count"])
                    if evaluation.get("useful_disagreement_count") is not None
                    else None
                ),
                "agent_queried_frames": queried_frames,
                "sim_frames": sim_frames,
                "query_ratio": round(query_ratio, 4),
                "agent_fallback_rate": (
                    float(evaluation["agent_fallback_rate"])
                    if evaluation.get("agent_fallback_rate") is not None
                    else _optional_float(assist.get("agent_fallback_rate"))
                ),
                "agent_timeout_rate": _optional_float(assist.get("agent_timeout_rate")),
                "low_ttc_agent_cautious_rate": (
                    float(low_ttc_analysis["agent_cautious_rate"])
                    if low_ttc_analysis.get("agent_cautious_rate") is not None
                    else None
                ),
                "low_ttc_baseline_cautious_rate": (
                    float(low_ttc_analysis["baseline_cautious_rate"])
                    if low_ttc_analysis.get("baseline_cautious_rate") is not None
                    else None
                ),
                "assist_applied_frames": (
                    int(assist["assist_applied_frames"])
                    if assist.get("assist_applied_frames") is not None
                    else None
                ),
                "assist_agent_query_frames": (
                    int(assist["agent_query_frames"])
                    if assist.get("agent_query_frames") is not None
                    else None
                ),
                "assist_agent_query_rate": (
                    float(assist["agent_query_rate"])
                    if assist.get("agent_query_rate") is not None
                    else None
                ),
                "assist_intervention_rate": (
                    float(assist["assist_intervention_rate"])
                    if assist.get("assist_intervention_rate") is not None
                    else None
                ),
                "assist_query_rejection_rate": _optional_float(assist.get("agent_query_rejection_rate")),
                "safety_arbitration_rejection_rate": _optional_float(
                    assist.get("safety_arbitration_rejection_rate")
                ),
                "safety_arbitration_rejection_reason_counts": assist.get(
                    "safety_arbitration_rejection_reason_counts"
                ),
                "end_to_end_query_success_rate": _optional_float(
                    assist.get("end_to_end_query_success_rate")
                ),
                "assist_p50_api_call_ms": _optional_float(
                    assist_latency.get("p50_api_call_ms")
                    if assist_latency.get("p50_api_call_ms") is not None
                    else compare_latency.get("p50_compare_ms")
                ),
                "assist_p95_api_call_ms": _optional_float(
                    assist_latency.get("p95_api_call_ms")
                    if assist_latency.get("p95_api_call_ms") is not None
                    else compare_latency.get("p95_compare_ms")
                ),
                "assist_over_step_budget_rate": _optional_float(
                    assist_latency.get("over_step_budget_rate")
                    if assist_latency.get("over_step_budget_rate") is not None
                    else compare_latency.get("over_step_budget_rate")
                ),
                "stale_response_discard_rate": _optional_float(
                    assist.get("stale_response_discard_rate")
                    if assist.get("stale_response_discard_rate") is not None
                    else evaluation.get("stale_response_rate")
                ),
                "control_loop_p50_ms": _optional_float(control_loop_latency.get("p50_ms")),
                "control_loop_p95_ms": _optional_float(control_loop_latency.get("p95_ms")),
                "control_loop_over_budget_rate": _optional_float(
                    control_loop_latency.get("over_step_budget_rate")
                ),
                "lane_change_completion_time_s": _optional_float(
                    lane_change_maneuver.get("completion_time_s")
                ),
                "lane_change_completed": (
                    1.0 if lane_change_maneuver.get("completed") is True
                    else (0.0 if assist else None)
                ),
                "lane_change_failure_reason": lane_change_maneuver.get("failure_reason"),
                "assist_reject_reason_counts": assist.get("assist_reject_reason_counts"),
                "query_rejection_reason_counts": assist.get("query_rejection_reason_counts"),
                "assist_validation_status_counts": assist.get("agent_validation_status_counts"),
                "assist_fallback_reason_counts": assist.get("agent_fallback_reason_counts"),
                "assist_api_attempt_count_total": assist.get("agent_api_attempt_count_total"),
                "assist_api_attempt_count_max": assist.get("agent_api_attempt_count_max"),
                "assist_api_payload_variant_counts": assist.get(
                    "agent_api_payload_variant_counts"
                ),
                "post_lane_change_cruise_frames": assist.get(
                    "post_lane_change_cruise_frames"
                ),
                "post_lane_change_handoff_frames": assist.get(
                    "post_lane_change_handoff_frames"
                ),
                "assist_agent_intent_distribution": assist.get("agent_intent_distribution"),
            }
        )
    return rows


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _sample_statistics(values: List[float]) -> Dict[str, Any]:
    cleaned = [float(value) for value in values]
    count = len(cleaned)
    if count == 0:
        return {"n": 0, "mean": None, "sample_sd": None, "standard_error": None, "ci95": None}
    mean_value = statistics.fmean(cleaned)
    if count == 1:
        return {
            "n": 1,
            "mean": round(mean_value, 6),
            "sample_sd": None,
            "standard_error": None,
            "ci95": None,
        }
    sample_sd = statistics.stdev(cleaned)
    standard_error = sample_sd / math.sqrt(count)
    degrees_freedom = count - 1
    critical = T_CRITICAL_95.get(degrees_freedom, 1.96)
    margin = critical * standard_error
    return {
        "n": count,
        "mean": round(mean_value, 6),
        "sample_sd": round(sample_sd, 6),
        "standard_error": round(standard_error, 6),
        "ci95": [round(mean_value - margin, 6), round(mean_value + margin, 6)],
    }


def _grouped_statistics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case"]), []).append(row)
    result: Dict[str, Any] = {}
    for case, case_rows in sorted(grouped.items()):
        result[case] = {
            "num_runs": len(case_rows),
            "seeds": sorted(
                int(row["random_seed"])
                for row in case_rows
                if row.get("random_seed") is not None
            ),
            "metrics": {
                field: _sample_statistics(
                    [float(row[field]) for row in case_rows if row.get(field) is not None]
                )
                for field in STATISTIC_FIELDS
            },
        }
    return result


def _paired_ab_statistics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    paired: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    for row in rows:
        case = str(row["case"])
        if case.endswith("_baseline"):
            base_case, variant = case[: -len("_baseline")], "baseline"
        elif case.endswith("_assist"):
            base_case, variant = case[: -len("_assist")], "assist"
        else:
            continue
        seed = row.get("random_seed")
        if seed is None:
            continue
        paired.setdefault(base_case, {}).setdefault(variant, {})[int(seed)] = row

    result: Dict[str, Any] = {}
    for base_case, variants in sorted(paired.items()):
        baseline_by_seed = variants.get("baseline", {})
        assist_by_seed = variants.get("assist", {})
        common_seeds = sorted(set(baseline_by_seed) & set(assist_by_seed))
        if not common_seeds:
            continue
        result[base_case] = {
            "num_pairs": len(common_seeds),
            "paired_seeds": common_seeds,
            "difference_definition": "assist_minus_baseline",
            "non_pairable_metrics": {
                "lane_change_completion_time_s": (
                    "Baseline does not initiate the Agent lane-change maneuver; "
                    "use assist_only_statistics for completion rate and time."
                )
            },
            "metrics": {
                field: _sample_statistics(
                    [
                        float(assist_by_seed[seed][field]) - float(baseline_by_seed[seed][field])
                        for seed in common_seeds
                        if assist_by_seed[seed].get(field) is not None
                        and baseline_by_seed[seed].get(field) is not None
                    ]
                )
                for field in STATISTIC_FIELDS
            },
        }
    return result


def _assist_only_statistics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize Agent maneuver completion without inventing a baseline time."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        case = str(row.get("case", ""))
        if not case.endswith("_assist"):
            continue
        grouped.setdefault(case[: -len("_assist")], []).append(row)

    result: Dict[str, Any] = {}
    for base_case, case_rows in sorted(grouped.items()):
        completed_rows = [row for row in case_rows if row.get("lane_change_completed") == 1.0]
        failure_counts: Dict[str, int] = {}
        for row in case_rows:
            if row.get("lane_change_completed") == 1.0:
                continue
            reason = str(row.get("lane_change_failure_reason") or "unspecified")
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
        result[base_case] = {
            "num_assist_runs": len(case_rows),
            "seeds": sorted(
                int(row["random_seed"])
                for row in case_rows
                if row.get("random_seed") is not None
            ),
            "completed_runs": len(completed_rows),
            "completion_rate": round(len(completed_rows) / len(case_rows), 6)
            if case_rows else None,
            "successful_completion_time_s": _sample_statistics(
                [
                    float(row["lane_change_completion_time_s"])
                    for row in completed_rows
                    if row.get("lane_change_completion_time_s") is not None
                ]
            ),
            "failure_reason_counts": failure_counts,
        }
    return result


def main() -> int:
    args = _parse_args()
    report_root = Path(args.report_root)
    rows = _load_rows(report_root, str(args.run_glob))
    if not rows:
        raise SystemExit(
            f"No stress runs found under {report_root} matching run_glob={args.run_glob!r}"
        )

    overall = {
        "num_runs": len(rows),
        "collision_free_runs": sum(
            1 for row in rows if row["collision_episode_count"] == 0
        ),
        "total_collision_episodes": sum(row["collision_episode_count"] for row in rows),
        "total_collision_sensor_events": sum(
            row["collision_sensor_event_count"] for row in rows
        ),
        "total_legal_lane_crossings": sum(
            row["legal_lane_crossing_count"] for row in rows
        ),
        "total_illegal_lane_invasions": sum(
            row["illegal_lane_invasion_count"] for row in rows
        ),
        "total_low_ttc_frames": sum(row["low_ttc_frames"] for row in rows),
        "total_useful_disagreement_count": sum(
            row["useful_disagreement_count"]
            for row in rows
            if row["useful_disagreement_count"] is not None
        ),
        "mean_route_progress_m": _mean(
            [row["route_progress_m"] for row in rows if row["route_progress_m"] is not None]
        ),
        "mean_distance_traveled_m": _mean(
            [row["distance_traveled_m"] for row in rows if row["distance_traveled_m"] is not None]
        ),
        "mean_agreement_rate": _mean(
            [row["agreement_rate"] for row in rows if row["agreement_rate"] is not None]
        ),
        "mean_query_ratio": _mean([row["query_ratio"] for row in rows]),
        "mean_agent_fallback_rate": _mean(
            [row["agent_fallback_rate"] for row in rows if row["agent_fallback_rate"] is not None]
        ),
        "mean_agent_timeout_rate": _mean(
            [row["agent_timeout_rate"] for row in rows if row["agent_timeout_rate"] is not None]
        ),
        "mean_safety_arbitration_rejection_rate": _mean(
            [
                row["safety_arbitration_rejection_rate"]
                for row in rows
                if row["safety_arbitration_rejection_rate"] is not None
            ]
        ),
        "mean_low_ttc_agent_cautious_rate": _mean(
            [
                row["low_ttc_agent_cautious_rate"]
                for row in rows
                if row["low_ttc_agent_cautious_rate"] is not None
            ]
        ),
        "total_assist_applied_frames": sum(
            row["assist_applied_frames"]
            for row in rows
            if row["assist_applied_frames"] is not None
        ),
        "total_assist_agent_query_frames": sum(
            row["assist_agent_query_frames"]
            for row in rows
            if row["assist_agent_query_frames"] is not None
        ),
    }
    summary = {
        "schema_version": "stage10_stress_tables_summary_v4",
        "report_root": str(report_root),
        "run_glob": str(args.run_glob),
        "table2_rows": [
            {
                "case": row["case"],
                "random_seed": row["random_seed"],
                "frames": row["frames"],
                "route_progress_m": row["route_progress_m"],
                "route_completion_rate": row["route_completion_rate"],
                "distance_traveled_m": row["distance_traveled_m"],
                "collision_count": row["collision_count"],
                "collision_episode_count": row["collision_episode_count"],
                "collision_sensor_event_count": row["collision_sensor_event_count"],
                "lane_invasion_count": row["lane_invasion_count"],
                "legal_lane_crossing_count": row["legal_lane_crossing_count"],
                "illegal_lane_invasion_count": row["illegal_lane_invasion_count"],
                "maneuver_illegal_lane_invasion_count": row[
                    "maneuver_illegal_lane_invasion_count"
                ],
                "post_maneuver_illegal_lane_invasion_count": row[
                    "post_maneuver_illegal_lane_invasion_count"
                ],
                "unknown_lane_crossing_count": row["unknown_lane_crossing_count"],
                "offroad_rate": row["offroad_rate"],
                "mean_abs_longitudinal_jerk_mps3": row["mean_abs_longitudinal_jerk_mps3"],
                "max_abs_longitudinal_jerk_mps3": row["max_abs_longitudinal_jerk_mps3"],
                "episode_duration_s": row["episode_duration_s"],
                "maneuver_duration_s": row["maneuver_duration_s"],
                "low_ttc_frames": row["low_ttc_frames"],
                "safety_outcome": row["safety_outcome"],
            }
            for row in rows
        ],
        "table3_rows": [
            {
                "case": row["case"],
                "random_seed": row["random_seed"],
                "agreement_rate": row["agreement_rate"],
                "disagreement_rate": row["disagreement_rate"],
                "useful_disagreement_count": row["useful_disagreement_count"],
                "agent_queried_frames": row["agent_queried_frames"],
                "sim_frames": row["sim_frames"],
                "query_ratio": row["query_ratio"],
                "agent_fallback_rate": row["agent_fallback_rate"],
                "agent_timeout_rate": row["agent_timeout_rate"],
                "low_ttc_agent_cautious_rate": row["low_ttc_agent_cautious_rate"],
                "low_ttc_baseline_cautious_rate": row["low_ttc_baseline_cautious_rate"],
                "assist_applied_frames": row["assist_applied_frames"],
                "assist_agent_query_frames": row["assist_agent_query_frames"],
                "assist_agent_query_rate": row["assist_agent_query_rate"],
                "assist_intervention_rate": row["assist_intervention_rate"],
                "assist_query_rejection_rate": row["assist_query_rejection_rate"],
                "safety_arbitration_rejection_rate": row[
                    "safety_arbitration_rejection_rate"
                ],
                "safety_arbitration_rejection_reason_counts": row[
                    "safety_arbitration_rejection_reason_counts"
                ],
                "end_to_end_query_success_rate": row["end_to_end_query_success_rate"],
                "assist_p50_api_call_ms": row["assist_p50_api_call_ms"],
                "assist_p95_api_call_ms": row["assist_p95_api_call_ms"],
                "assist_over_step_budget_rate": row["assist_over_step_budget_rate"],
                "stale_response_discard_rate": row["stale_response_discard_rate"],
                "control_loop_p50_ms": row["control_loop_p50_ms"],
                "control_loop_p95_ms": row["control_loop_p95_ms"],
                "control_loop_over_budget_rate": row["control_loop_over_budget_rate"],
                "lane_change_completion_time_s": row["lane_change_completion_time_s"],
                "lane_change_completed": row["lane_change_completed"],
                "lane_change_failure_reason": row["lane_change_failure_reason"],
                "assist_reject_reason_counts": row["assist_reject_reason_counts"],
                "query_rejection_reason_counts": row["query_rejection_reason_counts"],
                "assist_validation_status_counts": row["assist_validation_status_counts"],
                "assist_fallback_reason_counts": row["assist_fallback_reason_counts"],
                "assist_api_attempt_count_total": row["assist_api_attempt_count_total"],
                "assist_api_attempt_count_max": row["assist_api_attempt_count_max"],
                "assist_api_payload_variant_counts": row[
                    "assist_api_payload_variant_counts"
                ],
                "post_lane_change_cruise_frames": row[
                    "post_lane_change_cruise_frames"
                ],
                "post_lane_change_handoff_frames": row[
                    "post_lane_change_handoff_frames"
                ],
                "assist_agent_intent_distribution": row["assist_agent_intent_distribution"],
            }
            for row in rows
        ],
        "overall": overall,
        "grouped_statistics": _grouped_statistics(rows),
        "paired_ab_statistics": _paired_ab_statistics(rows),
        "assist_only_statistics": _assist_only_statistics(rows),
    }

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== TABLE 2 ===")
    print(
        "case,seed,frames,route_progress_m,route_completion_rate,distance_traveled_m,"
        "collision_count,collision_episode_count,collision_sensor_event_count,"
        "lane_invasion_count,legal_lane_crossing_count,illegal_lane_invasion_count,"
        "unknown_lane_crossing_count,offroad_rate,mean_abs_jerk_mps3,"
        "max_abs_jerk_mps3,episode_duration_s,maneuver_duration_s,"
        "low_ttc_frames,safety_outcome"
    )
    for row in summary["table2_rows"]:
        print(
            f"{row['case']},{row['random_seed']},{row['frames']},{_fmt_float(row['route_progress_m'])},"
            f"{_fmt_float(row['route_completion_rate'])},{_fmt_float(row['distance_traveled_m'])},"
            f"{row['collision_count']},{row['collision_episode_count']},"
            f"{row['collision_sensor_event_count']},{row['lane_invasion_count']},"
            f"{row['legal_lane_crossing_count']},{row['illegal_lane_invasion_count']},"
            f"{row['unknown_lane_crossing_count']},{_fmt_float(row['offroad_rate'])},"
            f"{_fmt_float(row['mean_abs_longitudinal_jerk_mps3'])},"
            f"{_fmt_float(row['max_abs_longitudinal_jerk_mps3'])},"
            f"{_fmt_float(row['episode_duration_s'])},"
            f"{_fmt_float(row['maneuver_duration_s'])},"
            f"{row['low_ttc_frames']},{row['safety_outcome']}"
        )

    print("\n=== TABLE 3 ===")
    print(
        "case,seed,agreement_rate,disagreement_rate,useful_disagreement_count,"
        "agent_queried_frames,sim_frames,query_ratio,agent_fallback_rate,agent_timeout_rate,"
        "low_ttc_agent_cautious_rate,low_ttc_baseline_cautious_rate,"
        "assist_agent_query_frames,assist_agent_query_rate,"
        "assist_applied_frames,assist_intervention_rate,assist_query_rejection_rate,"
        "safety_arbitration_rejection_rate,end_to_end_query_success_rate,"
        "assist_p50_api_call_ms,assist_p95_api_call_ms,assist_over_step_budget_rate,"
        "stale_response_discard_rate,control_loop_p50_ms,control_loop_p95_ms,"
        "control_loop_over_budget_rate,"
        "lane_change_completion_time_s,lane_change_completed,lane_change_failure_reason,"
        "query_rejection_reason_counts,"
        "safety_arbitration_rejection_reason_counts,"
        "assist_validation_status_counts,assist_fallback_reason_counts,"
        "assist_api_attempt_count_total,assist_api_attempt_count_max,"
        "assist_api_payload_variant_counts"
    )
    for row in summary["table3_rows"]:
        useful_count = _fmt_int(row["useful_disagreement_count"])
        fallback_rate = _fmt_float(row["agent_fallback_rate"])
        timeout_rate = _fmt_float(row["agent_timeout_rate"])
        agent_cautious = _fmt_float(row["low_ttc_agent_cautious_rate"])
        baseline_cautious = _fmt_float(row["low_ttc_baseline_cautious_rate"])
        assist_queries = _fmt_int(row["assist_agent_query_frames"])
        assist_query_rate = _fmt_float(row["assist_agent_query_rate"])
        assist_applied = _fmt_int(row["assist_applied_frames"])
        assist_rate = _fmt_float(row["assist_intervention_rate"])
        reject_reasons = json.dumps(row["query_rejection_reason_counts"] or {}, sort_keys=True)
        arbitration_reasons = json.dumps(
            row["safety_arbitration_rejection_reason_counts"] or {}, sort_keys=True
        )
        validation_counts = json.dumps(row["assist_validation_status_counts"] or {}, sort_keys=True)
        fallback_reasons = json.dumps(row["assist_fallback_reason_counts"] or {}, sort_keys=True)
        payload_variants = json.dumps(
            row["assist_api_payload_variant_counts"] or {}, sort_keys=True
        )
        print(
            f"{row['case']},{row['random_seed']},{_fmt_float(row['agreement_rate'])},"
            f"{_fmt_float(row['disagreement_rate'])},"
            f"{useful_count},{row['agent_queried_frames']},"
            f"{row['sim_frames']},{row['query_ratio']:.4f},{fallback_rate},{timeout_rate},"
            f"{agent_cautious},{baseline_cautious},{assist_queries},{assist_query_rate},"
            f"{assist_applied},{assist_rate},{_fmt_float(row['assist_query_rejection_rate'])},"
            f"{_fmt_float(row['safety_arbitration_rejection_rate'])},"
            f"{_fmt_float(row['end_to_end_query_success_rate'])},"
            f"{_fmt_float(row['assist_p50_api_call_ms'])},{_fmt_float(row['assist_p95_api_call_ms'])},"
            f"{_fmt_float(row['assist_over_step_budget_rate'])},"
            f"{_fmt_float(row['stale_response_discard_rate'])},"
            f"{_fmt_float(row['control_loop_p50_ms'])},{_fmt_float(row['control_loop_p95_ms'])},"
            f"{_fmt_float(row['control_loop_over_budget_rate'])},"
            f"{_fmt_float(row['lane_change_completion_time_s'])},"
            f"{_fmt_float(row['lane_change_completed'])},{row['lane_change_failure_reason'] or ''},"
            f"{reject_reasons},{arbitration_reasons},"
            f"{validation_counts},{fallback_reasons},"
            f"{_fmt_int(row['assist_api_attempt_count_total'])},"
            f"{_fmt_int(row['assist_api_attempt_count_max'])},{payload_variants}"
        )

    print("\n=== OVERALL ===")
    print(json.dumps(overall, indent=2))
    print("\n=== GROUPED STATISTICS (mean, sample SD, SE, 95% t-CI) ===")
    print(json.dumps(summary["grouped_statistics"], indent=2))
    print("\n=== PAIRED A/B STATISTICS (assist minus baseline) ===")
    print(json.dumps(summary["paired_ab_statistics"], indent=2))
    print("\n=== ASSIST-ONLY MANEUVER STATISTICS ===")
    print(json.dumps(summary["assist_only_statistics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
