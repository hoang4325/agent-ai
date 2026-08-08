from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

from agent_ai.shared.artifact_io import append_jsonl as _append_jsonl
from agent_ai.shared.artifact_io import write_json as _write_json
from agent_ai.shared.logging_setup import configure_logging

from .evaluation import summarize_world_session
from .planner_interface import build_planner_interface_payload
from .cost_memory import CostMemory
from .interaction_predictor import apply_interaction_prediction
from .motion_predictor import annotate_tracks_with_prediction
from .prediction_loader import list_stage1_frame_artifacts, load_normalized_prediction
from agent_ai.authority.soft_contract import build_soft_contract_from_behavior
from .risk_engine import RiskAssessmentEngine
from .tactical_rules import RuleBasedTacticalPolicy
from .tracker import SimpleObjectTracker
from .visualization import save_world_visualization_bundle
from .builder import WorldStateBuilder

LOGGER = logging.getLogger("agent_ai.world_state.replay_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Stage 2 on top of existing Stage 1 session artifacts.")
    parser.add_argument("--stage1-session", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prediction-variant",
        choices=("auto", "baseline_zero_bev", "bridge_minimal"),
        default="auto",
    )
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-missed-frames", type=int, default=2)
    parser.add_argument("--vehicle-match-distance-m", type=float, default=5.0)
    parser.add_argument("--vru-match-distance-m", type=float, default=3.0)
    parser.add_argument("--static-match-distance-m", type=float, default=2.5)
    parser.add_argument("--cruise-speed-mps", type=float, default=8.0)
    return parser.parse_args()


def run_replay(args: argparse.Namespace) -> Dict[str, Any]:
    stage1_session_dir = Path(args.stage1_session)
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    artifacts = list_stage1_frame_artifacts(
        stage1_session_dir,
        prediction_variant=args.prediction_variant,
    )
    if args.max_frames and args.max_frames > 0:
        artifacts = artifacts[: args.max_frames]
    if not artifacts:
        raise RuntimeError(f"No infer-ready frames found in stage1 session: {stage1_session_dir}")

    tracker = SimpleObjectTracker(
        max_missed_frames=args.max_missed_frames,
        vehicle_match_distance_m=args.vehicle_match_distance_m,
        vru_match_distance_m=args.vru_match_distance_m,
        static_match_distance_m=args.static_match_distance_m,
    )
    risk_engine = RiskAssessmentEngine()
    world_builder = WorldStateBuilder()
    policy = RuleBasedTacticalPolicy(cruise_speed_mps=args.cruise_speed_mps)
    cost_memory = CostMemory()

    world_states = []
    decisions = []

    manifest = {
        "stage1_session": str(stage1_session_dir),
        "output_dir": str(output_dir),
        "prediction_variant": args.prediction_variant,
        "min_score": args.min_score,
        "num_artifacts": len(artifacts),
        "algorithm_tier": "P2",
    }
    _write_json(output_dir / "stage2_manifest.json", manifest)

    for artifact in artifacts:
        normalized = load_normalized_prediction(artifact, min_score=args.min_score)
        tracked_objects = tracker.update(normalized)
        annotate_tracks_with_prediction(tracked_objects)
        interaction = apply_interaction_prediction(
            tracked_objects,
            ego_speed_mps=float(normalized.ego.speed_mps),
            ensure_independent=False,
        )
        risk_summary = risk_engine.evaluate(
            tracked_objects,
            ego_speed_mps=float(normalized.ego.speed_mps),
            interaction=interaction,
        )
        world_state = world_builder.build(normalized, tracked_objects, risk_summary)
        # Stash interaction + adaptive weights into decision context for consumers.
        world_state.decision_context = {
            **world_state.decision_context,
            "interaction_tags": list(interaction.tags),
            "soft_weight_profile": [cost_memory.last_profile],
        }
        decision = policy.decide(world_state, cost_memory=cost_memory)
        cost_memory.observe(
            min_ttc_s=risk_summary.minimum_ttc_seconds,
            highest_risk=risk_summary.highest_risk_level,
            maneuver=decision.maneuver,
            interaction_severity=interaction.max_interaction_severity,
            object_count=len(tracked_objects),
            nearest_front_m=risk_summary.nearest_front_vehicle_distance_m,
        )
        soft_bundle = build_soft_contract_from_behavior(
            frame_id=int(world_state.frame_id),
            sim_time_s=float(world_state.timestamp),
            maneuver=decision.maneuver,
            ego_speed_mps=float(world_state.ego.speed_mps),
            target_speed_mps=float(decision.target_speed_mps),
            min_ttc_s=risk_summary.minimum_ttc_seconds,
            highest_risk=risk_summary.highest_risk_level,
            interaction_severity=float(interaction.max_interaction_severity),
            confidence=float(decision.confidence),
            front_gap_m=world_state.scene.front_free_space_m,
            reasoning_tags=decision.reasoning_tags,
        )
        planner_payload = build_planner_interface_payload(
            world_state,
            decision,
            soft_contract=soft_bundle.to_dict(),
            interaction_summary=interaction.to_dict(),
            cost_memory_snapshot=cost_memory.snapshot(),
        )

        frame_output_dir = frames_dir / artifact.sample_name
        _write_json(frame_output_dir / "normalized_prediction.json", normalized.to_dict())
        _write_json(
            frame_output_dir / "tracked_objects.json",
            {
                "sample_name": normalized.sample_name,
                "frame_id": normalized.frame_id,
                "tracked_objects": [item.to_dict() for item in tracked_objects],
                "interaction_summary": interaction.to_dict(),
            },
        )
        _write_json(frame_output_dir / "scene_summary.json", world_state.scene.to_dict())
        _write_json(frame_output_dir / "risk_summary.json", world_state.risk_summary.to_dict())
        _write_json(frame_output_dir / "world_state.json", world_state.to_dict())
        _write_json(frame_output_dir / "decision_intent.json", decision.to_dict())
        _write_json(frame_output_dir / "soft_contract.json", soft_bundle.to_dict())
        _write_json(frame_output_dir / "planner_interface_payload.json", planner_payload.to_dict())

        _append_jsonl(
            output_dir / "decision_timeline.jsonl",
            {
                "sample_name": normalized.sample_name,
                "frame_id": normalized.frame_id,
                "timestamp": normalized.timestamp,
                "decision": decision.to_dict(),
                "highest_risk_level": world_state.risk_summary.highest_risk_level,
                "front_free_space_m": world_state.scene.front_free_space_m,
                "active_object_count": world_state.scene.active_object_count,
            },
        )
        _append_jsonl(
            output_dir / "world_state_timeline.jsonl",
            {
                "sample_name": normalized.sample_name,
                "frame_id": normalized.frame_id,
                "timestamp": normalized.timestamp,
                "world_state": world_state.to_dict(),
            },
        )

        world_states.append(world_state)
        decisions.append(decision)
        LOGGER.info(
            "Stage2 replay frame=%d sample=%s objects=%d maneuver=%s highest_risk=%s",
            normalized.frame_id,
            normalized.sample_name,
            len(tracked_objects),
            decision.maneuver,
            world_state.risk_summary.highest_risk_level,
        )

    evaluation_summary = summarize_world_session(world_states, decisions)
    _write_json(output_dir / "evaluation_summary.json", evaluation_summary)
    save_world_visualization_bundle(
        output_root=output_dir / "visualization",
        world_states=world_states,
        decisions=decisions,
    )
    LOGGER.info("Stage2 replay complete output_dir=%s", output_dir)
    return evaluation_summary


def main() -> None:
    args = parse_args()
    configure_logging()
    run_replay(args)


if __name__ == "__main__":
    main()
