"""
Lightweight semantic algorithm cases (B3) + regression harness (B1).

Cases live under agent_ai/benchmark/semantic_cases/v1/*.json and describe
synthetic ego/tracks/lane context with expected / forbidden outcomes.

No CARLA required. Suitable for CI.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from agent_ai.behavior.lane.lane_change_cost import evaluate_lane_change_candidates
from agent_ai.behavior.lane.schema import JunctionContext, LaneAwareObject, LaneContext, LaneDescriptor
from agent_ai.world_state.frenet import default_straight_corridor
from agent_ai.world_state.interaction_predictor import apply_interaction_prediction
from agent_ai.world_state.motion_predictor import annotate_tracks_with_prediction
from agent_ai.world_state.risk_engine import RiskAssessmentEngine
from agent_ai.world_state.schema import (
    EgoState,
    RiskSummary,
    SceneSummary,
    TrackedObject,
    WorldState,
)
from agent_ai.world_state.tactical_rules import RuleBasedTacticalPolicy

CASES_DIR = Path(__file__).resolve().parent / "semantic_cases" / "v1"


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failures: List[str] = field(default_factory=list)
    observations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "failures": list(self.failures),
            "observations": dict(self.observations),
        }


def default_cases_dir() -> Path:
    return CASES_DIR


def list_semantic_cases(cases_dir: str | Path | None = None) -> List[Path]:
    root = Path(cases_dir) if cases_dir else CASES_DIR
    if not root.exists():
        return []
    return sorted(root.glob("*.json"))


def load_case(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ego(speed: float, frame_id: int = 0) -> EgoState:
    return EgoState(
        frame_id=frame_id,
        timestamp=0.0,
        sample_name="semantic",
        town="TownSynthetic",
        weather={},
        position_world=[0.0, 0.0, 0.0],
        velocity_world=[speed, 0.0, 0.0],
        speed_mps=speed,
        yaw_deg=0.0,
        route_progress_m=0.0,
        world_from_ego_bevfusion=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def _track_from_dict(raw: Dict[str, Any]) -> TrackedObject:
    pos = list(raw.get("position_ego") or [0.0, 0.0, 0.0])
    while len(pos) < 3:
        pos.append(0.0)
    vel = list(raw.get("velocity_ego") or [0.0, 0.0])
    while len(vel) < 2:
        vel.append(0.0)
    x, y = float(pos[0]), float(pos[1])
    vx, vy = float(vel[0]), float(vel[1])
    dist = math.hypot(x, y)
    class_group = str(raw.get("class_group") or "vehicle")
    return TrackedObject(
        track_id=int(raw.get("track_id") or 0),
        class_id=int(raw.get("class_id") or 0),
        class_name=str(raw.get("class_name") or class_group),
        class_group=class_group,
        latest_detection_id=str(raw.get("detection_id") or f"d{raw.get('track_id', 0)}"),
        age_frames=int(raw.get("age_frames") or 5),
        hits=int(raw.get("hits") or 5),
        missed_frames=int(raw.get("missed_frames") or 0),
        is_occluded_est=bool(raw.get("is_occluded_est") or False),
        score=float(raw.get("score") or 0.9),
        mean_score=float(raw.get("mean_score") or 0.9),
        position_ego=pos[:3],
        velocity_ego=vel[:2],
        speed_mps=math.hypot(vx, vy),
        bbox=[x, y, 0.0, 4.0, 2.0, 1.5, 0.0, vx, vy],
        size_xyz=[4.0, 2.0, 1.5],
        yaw_rad=float(raw.get("yaw_rad") or 0.0),
        distance_m=dist,
        bearing_deg=math.degrees(math.atan2(y, x)) if dist > 1e-6 else 0.0,
        ttc_seconds=raw.get("ttc_seconds"),
        relative_sector=str(raw.get("relative_sector") or ("front" if x >= 0 else "rear")),
        source_confidence=float(raw.get("source_confidence") or 0.9),
    )


def _lane_descriptor(raw: Dict[str, Any] | None, role: str, *, default_exists: bool = True) -> LaneDescriptor:
    data = raw or {}
    exists = bool(data.get("exists", default_exists))
    return LaneDescriptor(
        exists=exists,
        role=str(data.get("role") or role),
        road_id=data.get("road_id", 1 if exists else None),
        section_id=data.get("section_id", 0 if exists else None),
        lane_id=data.get("lane_id", 1 if exists else None),
        lane_width_m=data.get("lane_width_m", 3.5 if exists else None),
        lane_type=data.get("lane_type", "Driving" if exists else None),
        lane_change=data.get("lane_change", "Both" if exists else None),
        same_direction_as_ego=bool(data.get("same_direction_as_ego", True)),
        transform_carla=data.get("transform_carla"),
        transform_bevfusion_world=data.get("transform_bevfusion_world"),
        lane_change_allowed_from_current=bool(data.get("lane_change_allowed_from_current", exists)),
    )


def _lane_context_from_case(case: Dict[str, Any]) -> LaneContext:
    lc = case.get("lane_context") or {}
    jc = lc.get("junction_context") or {}
    return LaneContext(
        frame_id=0,
        sample_name=str(case.get("case_id") or "semantic"),
        current_lane=_lane_descriptor(lc.get("current_lane"), "current", default_exists=True),
        left_lane=_lane_descriptor(lc.get("left_lane"), "left", default_exists=True),
        right_lane=_lane_descriptor(lc.get("right_lane"), "right", default_exists=True),
        forward_corridor=dict(lc.get("forward_corridor") or {"chain": [{"distance_m": float(i) * 5.0} for i in range(13)]}),
        junction_context=JunctionContext(
            is_in_junction=bool(jc.get("is_in_junction", False)),
            junction_ahead=bool(jc.get("junction_ahead", False)),
            distance_to_junction_m=jc.get("distance_to_junction_m"),
            branch_count_ahead=int(jc.get("branch_count_ahead") or 0),
            possible_turn_like_options=list(jc.get("possible_turn_like_options") or []),
            branch_distance_m=jc.get("branch_distance_m"),
        ),
    )


def _lane_objects_from_case(case: Dict[str, Any]) -> List[LaneAwareObject]:
    out: List[LaneAwareObject] = []
    for raw in case.get("lane_objects") or []:
        out.append(
            LaneAwareObject(
                track_id=int(raw.get("track_id") or 0),
                class_name=str(raw.get("class_name") or "car"),
                class_group=str(raw.get("class_group") or "vehicle"),
                position_world_carla=list(raw.get("position_world_carla") or [0.0, 0.0, 0.0]),
                lane_relation=str(raw.get("lane_relation") or "current_lane"),
                lane_tag=str(raw.get("lane_tag") or raw.get("lane_relation") or "current_lane"),
                object_lane_id=raw.get("object_lane_id"),
                object_road_id=raw.get("object_road_id"),
                longitudinal_m=raw.get("longitudinal_m"),
                lateral_m=raw.get("lateral_m"),
                is_front_in_current_lane=bool(raw.get("is_front_in_current_lane", False)),
                is_rear_in_current_lane=bool(raw.get("is_rear_in_current_lane", False)),
                is_blocking_current_lane=bool(raw.get("is_blocking_current_lane", False)),
                same_direction_as_ego_lane=bool(raw.get("same_direction_as_ego_lane", True)),
                distance_to_lane_center_m=raw.get("distance_to_lane_center_m"),
                source_track=dict(raw.get("source_track") or {}),
            )
        )
    return out


def _build_world(
    *,
    case: Dict[str, Any],
    tracks: List[TrackedObject],
    risk: RiskSummary,
) -> WorldState:
    ego_speed = float((case.get("ego") or {}).get("speed_mps") or 5.0)
    scene_raw = case.get("scene") or {}
    front_free = scene_raw.get("front_free_space_m")
    if front_free is None:
        front_tracks = [t for t in tracks if t.position_ego[0] > 0 and abs(t.position_ego[1]) <= 2.5]
        front_free = min((t.distance_m for t in front_tracks), default=None)
    nearest_front = None
    for t in tracks:
        if t.class_group == "vehicle" and t.position_ego[0] > 0:
            if nearest_front is None or t.distance_m < nearest_front["distance_m"]:
                nearest_front = {
                    "track_id": t.track_id,
                    "distance_m": t.distance_m,
                    "speed_mps": t.speed_mps,
                    "velocity_ego": list(t.velocity_ego),
                }
    nearest_vru = None
    for t in tracks:
        if t.class_group == "vru":
            if nearest_vru is None or t.distance_m < nearest_vru["distance_m"]:
                nearest_vru = {"track_id": t.track_id, "distance_m": t.distance_m}

    dc = case.get("decision_context") or {}
    return WorldState(
        frame_id=0,
        timestamp=0.0,
        sample_name=str(case.get("case_id") or "semantic"),
        sequence_index=0,
        ego=_ego(ego_speed),
        objects=tracks,
        scene=SceneSummary(
            active_object_count=len(tracks),
            front_free_space_m=None if front_free is None else float(front_free),
            left_side_occupancy=float(scene_raw.get("left_side_occupancy", 0.2)),
            right_side_occupancy=float(scene_raw.get("right_side_occupancy", 0.5)),
            rear_gap_m=scene_raw.get("rear_gap_m"),
            nearest_front_vehicle=nearest_front,
            nearest_vru=nearest_vru,
            nearest_any_object=nearest_front or nearest_vru,
            abnormal_flags=list(scene_raw.get("abnormal_flags") or []),
        ),
        risk_summary=risk,
        decision_context={
            "hard_constraints": list(dc.get("hard_constraints") or []),
            "soft_constraints": list(dc.get("soft_constraints") or []),
            "recommended_maneuvers": list(dc.get("recommended_maneuvers") or []),
        },
    )


def run_semantic_case(case: Dict[str, Any]) -> CaseResult:
    case_id = str(case.get("case_id") or "unknown")
    ego_speed = float((case.get("ego") or {}).get("speed_mps") or 5.0)
    tracks = [_track_from_dict(t) for t in case.get("tracks") or []]
    poly = default_straight_corridor(horizon_m=60.0)
    annotate_tracks_with_prediction(tracks, reference_polyline=poly)
    interaction = apply_interaction_prediction(tracks, ego_speed_mps=ego_speed, ensure_independent=False)
    risk_engine = RiskAssessmentEngine()
    risk = risk_engine.evaluate(tracks, ego_speed_mps=ego_speed, interaction=interaction)
    world = _build_world(case=case, tracks=tracks, risk=risk)
    policy = RuleBasedTacticalPolicy(cruise_speed_mps=float((case.get("ego") or {}).get("cruise_speed_mps") or 8.0))
    decision = policy.decide(world)

    lane_ctx = _lane_context_from_case(case)
    lane_objects = _lane_objects_from_case(case)
    # Attach source_track prediction fields for LC cost.
    by_id = {t.track_id: t for t in tracks}
    for obj in lane_objects:
        if obj.track_id in by_id and not obj.source_track:
            obj.source_track = by_id[obj.track_id].to_dict()

    left_ok = bool((case.get("lane_change_permission") or {}).get("left", True))
    right_ok = bool((case.get("lane_change_permission") or {}).get("right", True))
    if lane_ctx.junction_context.is_in_junction or (
        lane_ctx.junction_context.distance_to_junction_m is not None
        and float(lane_ctx.junction_context.distance_to_junction_m) < 15.0
    ):
        # Mirror validator junction gate for permission flags when not explicit.
        if "lane_change_permission" not in case:
            left_ok = False
            right_ok = False

    cost_eval = evaluate_lane_change_candidates(
        lane_context=lane_ctx,
        lane_objects=lane_objects,
        ego_speed_mps=ego_speed,
        left_ok=left_ok and bool(lane_ctx.left_lane.exists),
        left_reason=None if left_ok else "left_blocked",
        right_ok=right_ok and bool(lane_ctx.right_lane.exists),
        right_reason=None if right_ok else "right_blocked",
        route_prefer=(case.get("route") or {}).get("prefer"),
        current_front_gap_m=world.scene.front_free_space_m,
        left_occupancy=world.scene.left_side_occupancy,
        right_occupancy=world.scene.right_side_occupancy,
        highest_risk=risk.highest_risk_level,
    )

    mode_ids = sorted({m.get("mode_id") for t in tracks for m in (t.predicted_modes or []) if m.get("mode_id")})
    observations = {
        "risk_level": risk.highest_risk_level,
        "risk_score": risk.highest_risk_score,
        "min_ttc": risk.minimum_ttc_seconds,
        "maneuver": decision.maneuver,
        "target_speed_mps": decision.target_speed_mps,
        "reasoning_tags": list(decision.reasoning_tags),
        "track_tags": sorted({tag for t in tracks for tag in (t.reasoning_tags or [])}),
        "interaction_severity": interaction.max_interaction_severity,
        "lc_stage": cost_eval.get("stage"),
        "lc_selected": cost_eval.get("selected_maneuver"),
        "prediction_mode_ids": mode_ids,
        "flags": list(risk.flags),
    }

    expect = case.get("expect") or {}
    failures: List[str] = []

    def _check_any(key: str, observed: str | None, options: Sequence[str] | None) -> None:
        if not options:
            return
        if observed is None or observed not in set(options):
            failures.append(f"{key}: got {observed!r}, expected any of {list(options)}")

    def _check_forbidden(key: str, observed: str | None, forbidden: Sequence[str] | None) -> None:
        if not forbidden or observed is None:
            return
        if observed in set(forbidden):
            failures.append(f"{key}: got forbidden value {observed!r}")

    _check_any("risk_level", risk.highest_risk_level, expect.get("risk_levels_any"))
    _check_any("maneuver", decision.maneuver, expect.get("maneuvers_any"))
    _check_forbidden("maneuver", decision.maneuver, expect.get("maneuvers_forbidden"))
    _check_any("lc_stage", str(cost_eval.get("stage")), expect.get("lc_stages_any"))
    _check_forbidden("lc_stage", str(cost_eval.get("stage")), expect.get("lc_stages_forbidden"))
    _check_any("lc_selected", str(cost_eval.get("selected_maneuver")), expect.get("lc_selected_any"))
    _check_forbidden("lc_selected", str(cost_eval.get("selected_maneuver")), expect.get("lc_selected_forbidden"))

    tags_any = expect.get("tags_any") or []
    if tags_any:
        bag = set(observations["reasoning_tags"]) | set(observations["track_tags"]) | set(observations["flags"])
        if not any(t in bag for t in tags_any):
            failures.append(f"tags_any: none of {tags_any} found in {sorted(bag)}")

    modes_any = expect.get("prediction_modes_any") or []
    if modes_any:
        if not any(m in mode_ids for m in modes_any):
            failures.append(f"prediction_modes_any: none of {modes_any} in {mode_ids}")

    if expect.get("min_ttc_max") is not None and risk.minimum_ttc_seconds is not None:
        if float(risk.minimum_ttc_seconds) > float(expect["min_ttc_max"]):
            failures.append(
                f"min_ttc_max: ttc={risk.minimum_ttc_seconds} > {expect['min_ttc_max']}"
            )

    return CaseResult(
        case_id=case_id,
        passed=not failures,
        failures=failures,
        observations=observations,
    )


def run_all_semantic_cases(cases_dir: str | Path | None = None) -> List[CaseResult]:
    results: List[CaseResult] = []
    for path in list_semantic_cases(cases_dir):
        case = load_case(path)
        results.append(run_semantic_case(case))
    return results


def summarize_results(results: Sequence[CaseResult]) -> Dict[str, Any]:
    passed = sum(1 for r in results if r.passed)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "case_ids_failed": [r.case_id for r in results if not r.passed],
        "results": [r.to_dict() for r in results],
    }


def main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run semantic algorithm cases (B1/B3).")
    parser.add_argument("--cases-dir", default=str(CASES_DIR))
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    results = run_all_semantic_cases(args.cases_dir)
    summary = summarize_results(results)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
