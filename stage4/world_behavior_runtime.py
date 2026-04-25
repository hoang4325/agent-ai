from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from stage2.planner_interface import build_planner_interface_payload
from stage2.risk_engine import RiskAssessmentEngine
from stage2.tactical_rules import RuleBasedTacticalPolicy
from stage2.tracker import SimpleObjectTracker
from stage2.world_state_builder import WorldStateBuilder
from stage3.behavior_request_builder import build_behavior_request
from stage3.lane_reasoning import build_lane_relative_objects, summarize_lane_scene
from stage3.maneuver_validator import validate_maneuvers
from stage3.map_adapter import CarlaMapAdapter, bevfusion_world_xyz_to_carla
from stage3.schema import LaneAwareWorldState
from stage3b.behavior_request_v2_builder import build_behavior_request_v2
from stage3b.behavior_selector import select_behavior
from stage3b.route_context_builder import build_route_conditioned_scene, build_route_context
from stage3b.route_provider import load_route_manifest, resolve_route_hint

from .state_store import BehaviorFrameState, PerceptionFrameResult

LOGGER = logging.getLogger("stage4.world_behavior_runtime")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


class WorldBehaviorRuntime:
    def __init__(
        self,
        *,
        map_adapter: CarlaMapAdapter,
        output_dir: str | Path,
        route_option: str | None = None,
        preferred_lane: str | None = None,
        route_priority: str = "normal",
        route_manifest_path: str | Path | None = None,
        max_missed_frames: int = 2,
        vehicle_match_distance_m: float = 5.0,
        vru_match_distance_m: float = 3.0,
        static_match_distance_m: float = 2.5,
        cruise_speed_mps: float = 8.0,
        forward_horizon_m: float = 60.0,
        waypoint_step_m: float = 5.0,
    ) -> None:
        self.map_adapter = map_adapter
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.timeline_path = self.output_dir / "world_behavior_timeline.jsonl"
        self.route_option = route_option
        self.preferred_lane = preferred_lane
        self.route_priority = route_priority
        self.route_manifest = load_route_manifest(route_manifest_path)
        self.max_missed_frames = int(max_missed_frames)
        self.vehicle_match_distance_m = float(vehicle_match_distance_m)
        self.vru_match_distance_m = float(vru_match_distance_m)
        self.static_match_distance_m = float(static_match_distance_m)
        self.cruise_speed_mps = float(cruise_speed_mps)
        self.forward_horizon_m = float(forward_horizon_m)
        self.waypoint_step_m = float(waypoint_step_m)

        if self.timeline_path.exists():
            self.timeline_path.unlink()
        self.reset()

    def reset(self) -> None:
        self.tracker = SimpleObjectTracker(
            max_missed_frames=self.max_missed_frames,
            vehicle_match_distance_m=self.vehicle_match_distance_m,
            vru_match_distance_m=self.vru_match_distance_m,
            static_match_distance_m=self.static_match_distance_m,
        )
        self.risk_engine = RiskAssessmentEngine()
        self.world_builder = WorldStateBuilder()
        self.policy = RuleBasedTacticalPolicy(cruise_speed_mps=self.cruise_speed_mps)

    def process_prediction(self, perception_frame: PerceptionFrameResult) -> BehaviorFrameState:
        if perception_frame.raw_prediction is None:
            raise RuntimeError("Perception frame is missing raw_prediction; cannot run world/behavior update.")

        stage2_start = time.perf_counter()
        normalized = perception_frame.raw_prediction
        tracked_objects = self.tracker.update(normalized)
        risk_summary = self.risk_engine.evaluate(tracked_objects)
        world_state = self.world_builder.build(normalized, tracked_objects, risk_summary)
        decision_intent = self.policy.decide(world_state)
        planner_payload = build_planner_interface_payload(world_state, decision_intent)
        stage2_ms = (time.perf_counter() - stage2_start) * 1000.0

        stage3_start = time.perf_counter()
        lane_context = self.map_adapter.build_lane_context(
            frame_id=world_state.frame_id,
            sample_name=world_state.sample_name,
            ego_position_world_bevfusion=world_state.ego.position_world,
            horizon_m=self.forward_horizon_m,
            waypoint_step_m=self.waypoint_step_m,
        )
        lane_objects = build_lane_relative_objects(
            world_state=world_state.to_dict(),
            lane_context=lane_context,
            map_adapter=self.map_adapter,
        )
        lane_scene = summarize_lane_scene(lane_objects, lane_context)
        maneuver_validation = validate_maneuvers(
            world_state=world_state.to_dict(),
            stage2_decision_intent=decision_intent.to_dict(),
            lane_context=lane_context,
            lane_objects=lane_objects,
        )
        behavior_request = build_behavior_request(
            world_state=world_state.to_dict(),
            stage2_decision_intent=decision_intent.to_dict(),
            lane_context=lane_context,
            lane_objects=lane_objects,
            maneuver_validation=maneuver_validation,
        )
        ego_position_world_carla = bevfusion_world_xyz_to_carla(world_state.ego.position_world)
        lane_aware_world_state = LaneAwareWorldState(
            frame=int(world_state.frame_id),
            timestamp=float(world_state.timestamp),
            sample_name=str(world_state.sample_name),
            ego={
                **world_state.ego.to_dict(),
                "position_world_carla": ego_position_world_carla,
            },
            scene={
                **world_state.scene.to_dict(),
                "lane_scene": lane_scene,
            },
            risk_summary=world_state.risk_summary.to_dict(),
            decision_context={
                **world_state.decision_context,
                "stage2_decision_intent": decision_intent.to_dict(),
                "maneuver_validation_preview": maneuver_validation.to_dict(),
            },
            lane_context=lane_context.to_dict(),
            lane_relative_objects=[item.to_dict() for item in lane_objects],
            behavior_request_preview=behavior_request.to_dict(),
        )
        stage3_record = {
            "frame_id": int(world_state.frame_id),
            "timestamp": float(world_state.timestamp),
            "sample_name": str(world_state.sample_name),
            "stage2_decision": decision_intent.to_dict(),
            "planner_interface_payload": planner_payload.to_dict(),
            "lane_context": lane_context.to_dict(),
            "lane_scene": lane_scene,
            "lane_relative_objects": [item.to_dict() for item in lane_objects],
            "maneuver_validation": maneuver_validation.to_dict(),
            "behavior_request": behavior_request.to_dict(),
            "world_state": lane_aware_world_state.to_dict(),
        }
        route_hint = resolve_route_hint(
            stage3_record=stage3_record,
            cli_route_option=self.route_option,
            cli_preferred_lane=self.preferred_lane,
            cli_route_priority=self.route_priority,
            route_manifest=self.route_manifest,
        )
        route_context = build_route_context(stage3_record=stage3_record, route_hint=route_hint)
        route_conditioned_scene = build_route_conditioned_scene(
            stage3_record=stage3_record,
            route_context=route_context,
        )
        behavior_selection = select_behavior(
            stage3_record=stage3_record,
            route_context=route_context,
            route_conditioned_scene=route_conditioned_scene,
        )
        behavior_request_v2 = build_behavior_request_v2(
            stage3_record=stage3_record,
            route_context=route_context,
            route_conditioned_scene=route_conditioned_scene,
            behavior_selection=behavior_selection,
        )
        stage3_ms = (time.perf_counter() - stage3_start) * 1000.0

        frame_output_dir = self.frames_dir / perception_frame.sample_name
        _write_json(frame_output_dir / "perception_frame.json", perception_frame.to_dict())
        _write_json(frame_output_dir / "normalized_prediction.json", normalized.to_dict())
        _write_json(
            frame_output_dir / "tracked_objects.json",
            {
                "sample_name": normalized.sample_name,
                "frame_id": normalized.frame_id,
                "tracked_objects": [item.to_dict() for item in tracked_objects],
            },
        )
        _write_json(frame_output_dir / "risk_summary.json", world_state.risk_summary.to_dict())
        _write_json(frame_output_dir / "world_state.json", world_state.to_dict())
        _write_json(frame_output_dir / "decision_intent.json", decision_intent.to_dict())
        _write_json(frame_output_dir / "planner_interface_payload.json", planner_payload.to_dict())
        _write_json(frame_output_dir / "lane_context.json", lane_context.to_dict())
        _write_json(frame_output_dir / "lane_relative_objects.json", {"lane_relative_objects": [item.to_dict() for item in lane_objects]})
        _write_json(frame_output_dir / "maneuver_validation.json", maneuver_validation.to_dict())
        _write_json(frame_output_dir / "behavior_request.json", behavior_request.to_dict())
        _write_json(frame_output_dir / "lane_aware_world_state.json", lane_aware_world_state.to_dict())
        _write_json(frame_output_dir / "route_context.json", route_context.to_dict())
        _write_json(frame_output_dir / "route_conditioned_scene.json", route_conditioned_scene.to_dict())
        _write_json(frame_output_dir / "behavior_selection.json", behavior_selection.to_dict())
        _write_json(frame_output_dir / "behavior_request_v2.json", behavior_request_v2.to_dict())

        state = BehaviorFrameState(
            sample_name=perception_frame.sample_name,
            sequence_index=perception_frame.sequence_index,
            frame_id=perception_frame.frame_id,
            timestamp=perception_frame.timestamp,
            perception_latency_ms=perception_frame.perception_latency_ms,
            world_update_latency_ms=float(stage2_ms),
            behavior_latency_ms=float(stage3_ms),
            normalized_prediction=normalized.to_dict(),
            tracked_objects=[item.to_dict() for item in tracked_objects],
            risk_summary=world_state.risk_summary.to_dict(),
            world_state=world_state.to_dict(),
            decision_intent=decision_intent.to_dict(),
            planner_interface_payload=planner_payload.to_dict(),
            lane_context=lane_context.to_dict(),
            lane_scene=lane_scene,
            lane_relative_objects=[item.to_dict() for item in lane_objects],
            maneuver_validation=maneuver_validation.to_dict(),
            behavior_request=behavior_request.to_dict(),
            route_context=route_context.to_dict(),
            route_conditioned_scene=route_conditioned_scene.to_dict(),
            behavior_selection=behavior_selection.to_dict(),
            behavior_request_v2=behavior_request_v2.to_dict(),
            notes=[],
        )
        _append_jsonl(self.timeline_path, state.to_dict())
        LOGGER.info(
            "Stage4 world/behavior frame=%d sample=%s stage2_ms=%.1f stage3_ms=%.1f behavior=%s",
            state.frame_id,
            state.sample_name,
            state.world_update_latency_ms,
            state.behavior_latency_ms,
            state.behavior_request_v2["requested_behavior"],
        )
        return state
