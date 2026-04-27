from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _unique_lane_sequence(trace_rows: list[dict[str, Any]]) -> list[str]:
    sequence: list[str] = []
    for row in trace_rows:
        lane_id = str(row.get("ego_lane_id", "unknown"))
        if not sequence or sequence[-1] != lane_id:
            sequence.append(lane_id)
    return sequence


def _first_lane_transition(trace_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not trace_rows:
        return None
    initial_lane = str(trace_rows[0].get("ego_lane_id", "unknown"))
    for row in trace_rows:
        lane_id = str(row.get("ego_lane_id", "unknown"))
        if lane_id != initial_lane:
            return {
                "frame_idx": row.get("frame_idx"),
                "frame_id": row.get("frame_id"),
                "from_lane": initial_lane,
                "to_lane": lane_id,
                "route_progress_m": row.get("route_progress_m"),
                "ego_v_mps": row.get("ego_v_mps"),
            }
    return None


def _commit_frames(frame_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    commit_rows: list[dict[str, Any]] = []
    for row in frame_log:
        assist_request = row.get("assist_request") or {}
        intent = str(assist_request.get("tactical_intent") or row.get("agent_intent") or "")
        if row.get("assist_applied") and intent.startswith("commit_lane_change_"):
            commit_rows.append(
                {
                    "frame_idx": row.get("frame_idx"),
                    "frame_id": row.get("frame_id"),
                    "intent": intent,
                    "target_lane_id": assist_request.get("target_lane_id"),
                    "route_progress_m": row.get("route_progress_m"),
                    "continued": bool(row.get("assist_continued")),
                    "command": row.get("applied_command"),
                }
            )
    return commit_rows


def _collision_events(report_dir: Path) -> list[dict[str, Any]]:
    return _load_jsonl(report_dir / "collision_events.jsonl")


def analyze_run(report_dir: Path) -> dict[str, Any]:
    assist_path = report_dir / "stage10_agent_assist_evaluation.json"
    trace_path = report_dir / "ego_trace.jsonl"
    if not assist_path.exists():
        raise FileNotFoundError(f"Missing assist evaluation file: {assist_path}")
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing ego trace file: {trace_path}")

    assist = _load_json(assist_path)
    trace_rows = _load_jsonl(trace_path)
    frame_log = list(assist.get("frame_log") or [])
    commit_rows = _commit_frames(frame_log)
    first_transition = _first_lane_transition(trace_rows)
    lane_sequence = _unique_lane_sequence(trace_rows)
    collisions = _collision_events(report_dir)

    status = "no_commit"
    if commit_rows:
        status = "intent_only_commit"
    if first_transition is not None:
        status = "physical_lane_transition"

    return {
        "run": report_dir.name,
        "status": status,
        "lane_sequence": lane_sequence,
        "initial_lane": lane_sequence[0] if lane_sequence else None,
        "first_lane_transition": first_transition,
        "commit_intent_frames": len(commit_rows),
        "first_commit_intent": commit_rows[0] if commit_rows else None,
        "collision_events": len(collisions),
        "first_collision": collisions[0] if collisions else None,
        "assist_applied_frames": assist.get("assist_applied_frames"),
        "assist_query_frames": assist.get("agent_query_frames"),
        "validation_status_counts": assist.get("agent_validation_status_counts"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether a Stage10 assist run actually committed a lane change.")
    parser.add_argument("--report-root", required=True, help="Directory containing per-run report folders.")
    parser.add_argument(
        "--run-glob",
        default="*",
        help="Glob for selecting run directories under report-root. Defaults to all runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_root = Path(args.report_root)
    run_dirs = sorted(path for path in report_root.glob(args.run_glob) if path.is_dir())
    if not run_dirs:
        print(f"No run directories found under {report_root} matching {args.run_glob!r}")
        return 1

    for run_dir in run_dirs:
        try:
            summary = analyze_run(run_dir)
        except FileNotFoundError:
            continue
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
