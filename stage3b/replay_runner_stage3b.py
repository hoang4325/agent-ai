from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .behavior_request_v2_builder import build_behavior_request_v2
from .behavior_selector import select_behavior
from .evaluation_stage3b import summarize_stage3b_session
from .route_context_builder import build_route_conditioned_scene, build_route_context
from .route_provider import load_route_manifest, resolve_route_hint
from .visualization_stage3b import save_stage3b_visualization_bundle

LOGGER = logging.getLogger("stage3b.replay_runner")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 3A timeline: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _dedupe_stage3a_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for record in records:
        key = (int(record["frame_id"]), str(record["sample_name"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Stage 3B route-conditioned behavior layer on top of Stage 3A artifacts."
    )
    parser.add_argument("--stage3a-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--route-option", default=None)
    parser.add_argument("--preferred-lane", default=None)
    parser.add_argument("--route-priority", default="normal")
    parser.add_argument("--route-manifest", default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def run_replay(args: argparse.Namespace) -> Dict[str, Any]:
    stage3a_output_dir = Path(args.stage3a_output_dir)
    output_dir = Path(args.output_dir)
    timeline_path = stage3a_output_dir / "behavior_timeline.jsonl"
    stage3a_records = _dedupe_stage3a_records(_load_jsonl(timeline_path))
    if args.max_frames and args.max_frames > 0:
        stage3a_records = stage3a_records[: args.max_frames]
    if not stage3a_records:
        raise RuntimeError(f"No Stage 3A frames found in {stage3a_output_dir}")

    route_manifest = load_route_manifest(args.route_manifest)
    manifest = {
        "stage3a_output_dir": str(stage3a_output_dir),
        "output_dir": str(output_dir),
        "route_option": args.route_option,
        "preferred_lane": args.preferred_lane,
        "route_priority": args.route_priority,
        "route_manifest": args.route_manifest,
        "num_frames": int(len(stage3a_records)),
    }
    _write_json(output_dir / "stage3b_manifest.json", manifest)
    timeline_v2_path = output_dir / "behavior_timeline_v2.jsonl"
    if timeline_v2_path.exists():
        timeline_v2_path.unlink()

    output_frames_dir = output_dir / "frames"
    frame_records: List[Dict[str, Any]] = []

    for record in stage3a_records:
        route_hint = resolve_route_hint(
            stage3_record=record,
            cli_route_option=args.route_option,
            cli_preferred_lane=args.preferred_lane,
            cli_route_priority=args.route_priority,
            route_manifest=route_manifest,
        )
        route_context = build_route_context(stage3_record=record, route_hint=route_hint)
        route_conditioned_scene = build_route_conditioned_scene(
            stage3_record=record,
            route_context=route_context,
        )
        behavior_selection = select_behavior(
            stage3_record=record,
            route_context=route_context,
            route_conditioned_scene=route_conditioned_scene,
        )
        behavior_request_v2 = build_behavior_request_v2(
            stage3_record=record,
            route_context=route_context,
            route_conditioned_scene=route_conditioned_scene,
            behavior_selection=behavior_selection,
        )

        comparison = {
            "frame_id": int(record["frame_id"]),
            "sample_name": str(record["sample_name"]),
            "stage2_maneuver": str(record["stage2_decision"]["maneuver"]),
            "stage3a_behavior": str(record["behavior_request"]["requested_behavior"]),
            "stage3b_behavior": str(behavior_request_v2.requested_behavior),
            "route_option": str(route_context.route_option),
            "lane_change_stage": str(behavior_selection.lane_change_stage),
            "stage3b_overrode_stage3a": bool(
                record["behavior_request"]["requested_behavior"] != behavior_request_v2.requested_behavior
            ),
            "route_influenced": bool(behavior_selection.route_influence),
            "target_lane_changed": bool(
                str(record["behavior_request"].get("target_lane", "current"))
                != str(behavior_request_v2.target_lane)
            ),
        }

        frame_output_dir = output_frames_dir / str(record["sample_name"])
        _write_json(frame_output_dir / "route_context.json", route_context.to_dict())
        _write_json(frame_output_dir / "route_conditioned_scene.json", route_conditioned_scene.to_dict())
        _write_json(frame_output_dir / "behavior_selection.json", behavior_selection.to_dict())
        _write_json(frame_output_dir / "behavior_request_v2.json", behavior_request_v2.to_dict())
        _write_json(frame_output_dir / "stage3a_stage3b_comparison.json", comparison)

        timeline_record = {
            "frame_id": int(record["frame_id"]),
            "timestamp": float(record["timestamp"]),
            "sample_name": str(record["sample_name"]),
            "stage2_decision": dict(record["stage2_decision"]),
            "stage3a_behavior_request": dict(record["behavior_request"]),
            "maneuver_validation": dict(record["maneuver_validation"]),
            "lane_context": dict(record["lane_context"]),
            "lane_scene": dict(record["lane_scene"]),
            "lane_relative_objects": list(record["lane_relative_objects"]),
            "route_context": route_context.to_dict(),
            "route_conditioned_scene": route_conditioned_scene.to_dict(),
            "behavior_selection": behavior_selection.to_dict(),
            "behavior_request_v2": behavior_request_v2.to_dict(),
            "comparison": comparison,
            "world_state": dict(record["world_state"]),
        }
        _append_jsonl(timeline_v2_path, timeline_record)
        frame_records.append(timeline_record)

        LOGGER.info(
            "Stage3B frame=%d sample=%s route=%s stage3a=%s stage3b=%s lc_stage=%s",
            int(record["frame_id"]),
            str(record["sample_name"]),
            route_context.route_option,
            record["behavior_request"]["requested_behavior"],
            behavior_request_v2.requested_behavior,
            behavior_selection.lane_change_stage,
        )

    evaluation_summary = summarize_stage3b_session(frame_records)
    _write_json(output_dir / "evaluation_summary.json", evaluation_summary)
    save_stage3b_visualization_bundle(
        output_root=output_dir / "visualization",
        frame_records=frame_records,
    )
    LOGGER.info("Stage3B replay complete output_dir=%s", output_dir)
    return evaluation_summary


def main() -> None:
    args = parse_args()
    configure_logging()
    run_replay(args)


if __name__ == "__main__":
    main()
