from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .behavior_request_builder import build_behavior_request
from .evaluation_stage3 import summarize_stage3_session
from .lane_context_builder import build_lane_context
from .lane_reasoning import build_lane_relative_objects, summarize_lane_scene
from .maneuver_validator import validate_maneuvers
from .map_adapter import CarlaMapAdapter, bevfusion_world_xyz_to_carla
from .schema import LaneAwareWorldState
from .visualization_stage3 import save_stage3_visualization_bundle

LOGGER = logging.getLogger("stage3.replay_runner")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _list_stage2_frame_dirs(stage2_output_dir: Path) -> List[Path]:
    frames_root = stage2_output_dir / "frames"
    if not frames_root.exists():
        raise FileNotFoundError(f"Missing frames directory in Stage 2 output: {frames_root}")
    frame_dirs = [path for path in frames_root.iterdir() if path.is_dir()]
    frame_dirs.sort(key=lambda item: item.name)
    return frame_dirs


def _stage2_frame_bundle(frame_dir: Path) -> Dict[str, Any]:
    world_state = _load_json(frame_dir / "world_state.json")
    decision_intent = _load_json(frame_dir / "decision_intent.json")
    planner_payload = _load_json(frame_dir / "planner_interface_payload.json")
    return {
        "world_state": world_state,
        "decision_intent": decision_intent,
        "planner_interface_payload": planner_payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Stage 3A lane/route-aware behavior-scene layer on top of Stage 2 artifacts."
    )
    parser.add_argument("--stage2-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--carla-timeout-s", type=float, default=10.0)
    parser.add_argument("--load-town-if-needed", action="store_true")
    parser.add_argument("--forward-horizon-m", type=float, default=60.0)
    parser.add_argument("--waypoint-step-m", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def run_replay(args: argparse.Namespace) -> Dict[str, Any]:
    stage2_output_dir = Path(args.stage2_output_dir)
    output_dir = Path(args.output_dir)
    frame_dirs = _list_stage2_frame_dirs(stage2_output_dir)
    if args.max_frames and args.max_frames > 0:
        frame_dirs = frame_dirs[: args.max_frames]
    if not frame_dirs:
        raise RuntimeError(f"No Stage 2 frames found in {stage2_output_dir}")

    first_world_state = _load_json(frame_dirs[0] / "world_state.json")
    town = first_world_state["ego"].get("town")
    if not town:
        raise RuntimeError("Stage 2 world_state is missing ego.town, cannot align CARLA map context.")

    map_adapter = CarlaMapAdapter.connect(
        host=str(args.carla_host),
        port=int(args.carla_port),
        timeout_s=float(args.carla_timeout_s),
        town=str(town),
        load_town_if_needed=bool(args.load_town_if_needed),
    )

    manifest = {
        "stage2_output_dir": str(stage2_output_dir),
        "output_dir": str(output_dir),
        "carla_host": str(args.carla_host),
        "carla_port": int(args.carla_port),
        "town": str(town),
        "num_frames": int(len(frame_dirs)),
        "forward_horizon_m": float(args.forward_horizon_m),
        "waypoint_step_m": float(args.waypoint_step_m),
    }
    _write_json(output_dir / "stage3_manifest.json", manifest)

    frame_records: List[Dict[str, Any]] = []
    output_frames_dir = output_dir / "frames"

    for frame_dir in frame_dirs:
        bundle = _stage2_frame_bundle(frame_dir)
        world_state = bundle["world_state"]
        decision_intent = bundle["decision_intent"]
        planner_payload = bundle["planner_interface_payload"]

        lane_context = build_lane_context(
            world_state=world_state,
            map_adapter=map_adapter,
            forward_horizon_m=float(args.forward_horizon_m),
            waypoint_step_m=float(args.waypoint_step_m),
        )
        lane_objects = build_lane_relative_objects(
            world_state=world_state,
            lane_context=lane_context,
            map_adapter=map_adapter,
        )
        lane_scene = summarize_lane_scene(lane_objects, lane_context)
        maneuver_validation = validate_maneuvers(
            world_state=world_state,
            stage2_decision_intent=decision_intent,
            lane_context=lane_context,
            lane_objects=lane_objects,
        )
        behavior_request = build_behavior_request(
            world_state=world_state,
            stage2_decision_intent=decision_intent,
            lane_context=lane_context,
            lane_objects=lane_objects,
            maneuver_validation=maneuver_validation,
        )

        ego_position_world_carla = bevfusion_world_xyz_to_carla(world_state["ego"]["position_world"])
        lane_aware_world_state = LaneAwareWorldState(
            frame=int(world_state["frame"]),
            timestamp=float(world_state["timestamp"]),
            sample_name=str(world_state["sample_name"]),
            ego={
                **dict(world_state["ego"]),
                "position_world_carla": ego_position_world_carla,
            },
            scene={
                **dict(world_state["scene"]),
                "lane_scene": lane_scene,
            },
            risk_summary=dict(world_state["risk_summary"]),
            decision_context={
                **dict(world_state["decision_context"]),
                "stage2_decision_intent": dict(decision_intent),
                "maneuver_validation_preview": maneuver_validation.to_dict(),
            },
            lane_context=lane_context.to_dict(),
            lane_relative_objects=[item.to_dict() for item in lane_objects],
            behavior_request_preview=behavior_request.to_dict(),
        )

        comparison = {
            "frame_id": int(world_state["frame"]),
            "sample_name": str(world_state["sample_name"]),
            "stage2_maneuver": str(decision_intent["maneuver"]),
            "stage3_behavior_request": str(behavior_request.requested_behavior),
            "stage3_selected_maneuver": str(maneuver_validation.selected_maneuver),
            "stage3_overrode_stage2": bool(
                decision_intent["maneuver"] != maneuver_validation.selected_maneuver
            ),
            "current_lane_blocked": bool(lane_scene["current_lane_blocked"]),
            "junction_context": dict(lane_context.junction_context.to_dict()),
        }

        frame_output_dir = output_frames_dir / frame_dir.name
        _write_json(frame_output_dir / "lane_context.json", lane_context.to_dict())
        _write_json(
            frame_output_dir / "lane_relative_objects.json",
            {
                "frame": int(world_state["frame"]),
                "sample_name": str(world_state["sample_name"]),
                "lane_relative_objects": [item.to_dict() for item in lane_objects],
            },
        )
        _write_json(frame_output_dir / "maneuver_validation.json", maneuver_validation.to_dict())
        _write_json(frame_output_dir / "behavior_request.json", behavior_request.to_dict())
        _write_json(frame_output_dir / "lane_aware_world_state.json", lane_aware_world_state.to_dict())
        _write_json(frame_output_dir / "stage2_stage3_comparison.json", comparison)

        timeline_record = {
            "frame_id": int(world_state["frame"]),
            "timestamp": float(world_state["timestamp"]),
            "sample_name": str(world_state["sample_name"]),
            "stage2_decision": dict(decision_intent),
            "planner_interface_payload": dict(planner_payload),
            "lane_context": lane_context.to_dict(),
            "lane_scene": lane_scene,
            "lane_relative_objects": [item.to_dict() for item in lane_objects],
            "maneuver_validation": maneuver_validation.to_dict(),
            "behavior_request": behavior_request.to_dict(),
            "comparison": comparison,
            "world_state": lane_aware_world_state.to_dict(),
        }
        _append_jsonl(output_dir / "behavior_timeline.jsonl", timeline_record)
        frame_records.append(timeline_record)

        LOGGER.info(
            "Stage3A frame=%d sample=%s stage2=%s stage3=%s lane_blocked=%s junction=%s",
            int(world_state["frame"]),
            str(world_state["sample_name"]),
            decision_intent["maneuver"],
            behavior_request.requested_behavior,
            lane_scene["current_lane_blocked"],
            lane_context.junction_context.is_in_junction,
        )

    evaluation_summary = summarize_stage3_session(frame_records)
    _write_json(output_dir / "evaluation_summary.json", evaluation_summary)
    save_stage3_visualization_bundle(
        output_root=output_dir / "visualization",
        frame_records=frame_records,
    )
    LOGGER.info("Stage3A replay complete output_dir=%s", output_dir)
    return evaluation_summary


def main() -> None:
    args = parse_args()
    configure_logging()
    run_replay(args)


if __name__ == "__main__":
    main()
