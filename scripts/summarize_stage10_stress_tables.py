from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path
from typing import Any, Dict, List


RUN_SUFFIX_RE = re.compile(r"_run\d+$")


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


def _load_rows(report_root: Path, run_glob: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for driving_path in sorted(report_root.glob("*/stage10_driving_metrics.json")):
        run_dir = driving_path.parent
        if not fnmatch.fnmatch(run_dir.name, run_glob):
            continue
        evaluation_path = run_dir / "stage10_agent_live_evaluation.json"
        if not evaluation_path.exists():
            continue

        driving = json.loads(driving_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        low_ttc_analysis = evaluation.get("low_ttc_analysis") or {}
        queried_frames = int(evaluation.get("agent_queried_frames") or 0)
        sim_frames = int(evaluation.get("sim_frames") or 0)
        query_ratio = queried_frames / sim_frames if sim_frames else 0.0
        collision_count = int(driving.get("collision_count") or 0)
        safety_outcome = "Collision" if collision_count > 0 else "Collision-free"

        rows.append(
            {
                "run_name": run_dir.name,
                "case": _case_name(run_dir.name),
                "frames": sim_frames or int(driving.get("frames") or 0),
                "collision_count": collision_count,
                "low_ttc_frames": int(low_ttc_analysis.get("total_low_ttc_frames") or 0),
                "safety_outcome": safety_outcome,
                "agreement_rate": float(evaluation.get("agreement_rate") or 0.0),
                "disagreement_rate": float(evaluation.get("disagreement_rate") or 0.0),
                "useful_disagreement_count": int(evaluation.get("useful_disagreement_count") or 0),
                "agent_queried_frames": queried_frames,
                "sim_frames": sim_frames,
                "query_ratio": round(query_ratio, 4),
                "agent_fallback_rate": (
                    float(evaluation["agent_fallback_rate"])
                    if evaluation.get("agent_fallback_rate") is not None
                    else None
                ),
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
            }
        )
    return rows


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


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
        "collision_free_runs": sum(1 for row in rows if row["collision_count"] == 0),
        "total_low_ttc_frames": sum(row["low_ttc_frames"] for row in rows),
        "total_useful_disagreement_count": sum(row["useful_disagreement_count"] for row in rows),
        "mean_agreement_rate": _mean([row["agreement_rate"] for row in rows]),
        "mean_query_ratio": _mean([row["query_ratio"] for row in rows]),
        "mean_agent_fallback_rate": _mean(
            [row["agent_fallback_rate"] for row in rows if row["agent_fallback_rate"] is not None]
        ),
        "mean_low_ttc_agent_cautious_rate": _mean(
            [
                row["low_ttc_agent_cautious_rate"]
                for row in rows
                if row["low_ttc_agent_cautious_rate"] is not None
            ]
        ),
    }
    summary = {
        "schema_version": "stage10_stress_tables_summary_v1",
        "report_root": str(report_root),
        "run_glob": str(args.run_glob),
        "table2_rows": [
            {
                "case": row["case"],
                "frames": row["frames"],
                "collision_count": row["collision_count"],
                "low_ttc_frames": row["low_ttc_frames"],
                "safety_outcome": row["safety_outcome"],
            }
            for row in rows
        ],
        "table3_rows": [
            {
                "case": row["case"],
                "agreement_rate": row["agreement_rate"],
                "disagreement_rate": row["disagreement_rate"],
                "useful_disagreement_count": row["useful_disagreement_count"],
                "agent_queried_frames": row["agent_queried_frames"],
                "sim_frames": row["sim_frames"],
                "query_ratio": row["query_ratio"],
                "agent_fallback_rate": row["agent_fallback_rate"],
                "low_ttc_agent_cautious_rate": row["low_ttc_agent_cautious_rate"],
                "low_ttc_baseline_cautious_rate": row["low_ttc_baseline_cautious_rate"],
            }
            for row in rows
        ],
        "overall": overall,
    }

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== TABLE 2 ===")
    print("case,frames,collision_count,low_ttc_frames,safety_outcome")
    for row in summary["table2_rows"]:
        print(
            f"{row['case']},{row['frames']},{row['collision_count']},"
            f"{row['low_ttc_frames']},{row['safety_outcome']}"
        )

    print("\n=== TABLE 3 ===")
    print(
        "case,agreement_rate,disagreement_rate,useful_disagreement_count,"
        "agent_queried_frames,sim_frames,query_ratio,agent_fallback_rate,"
        "low_ttc_agent_cautious_rate,low_ttc_baseline_cautious_rate"
    )
    for row in summary["table3_rows"]:
        fallback_rate = "" if row["agent_fallback_rate"] is None else f"{row['agent_fallback_rate']:.4f}"
        agent_cautious = (
            "" if row["low_ttc_agent_cautious_rate"] is None
            else f"{row['low_ttc_agent_cautious_rate']:.4f}"
        )
        baseline_cautious = (
            "" if row["low_ttc_baseline_cautious_rate"] is None
            else f"{row['low_ttc_baseline_cautious_rate']:.4f}"
        )
        print(
            f"{row['case']},{row['agreement_rate']:.4f},{row['disagreement_rate']:.4f},"
            f"{row['useful_disagreement_count']},{row['agent_queried_frames']},"
            f"{row['sim_frames']},{row['query_ratio']:.4f},{fallback_rate},"
            f"{agent_cautious},{baseline_cautious}"
        )

    print("\n=== OVERALL ===")
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
