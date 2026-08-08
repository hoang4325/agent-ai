from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List

from .object_schema import DecisionIntent, WorldState


def summarize_stage2_session(
    world_states: Iterable[WorldState],
    decisions: Iterable[DecisionIntent],
) -> Dict[str, object]:
    world_states = list(world_states)
    decisions = list(decisions)
    maneuver_counts = Counter(decision.maneuver for decision in decisions)
    active_counts = [len(item.objects) for item in world_states]
    front_free_space_values = [
        item.scene.front_free_space_m
        for item in world_states
        if item.scene.front_free_space_m is not None
    ]
    track_lengths: Dict[int, int] = defaultdict(int)
    risk_counts = Counter(item.risk_summary.highest_risk_level for item in world_states)
    for world_state in world_states:
        for track in world_state.objects:
            track_lengths[int(track.track_id)] += 1

    mean_track_length = (
        float(sum(track_lengths.values()) / len(track_lengths)) if track_lengths else 0.0
    )
    stable_tracks = sum(1 for length in track_lengths.values() if length >= 3)
    return {
        "num_frames": int(len(world_states)),
        "num_decisions": int(len(decisions)),
        "unique_tracks": int(len(track_lengths)),
        "stable_tracks_3plus_frames": int(stable_tracks),
        "mean_track_length_frames": float(mean_track_length),
        "average_active_tracks_per_frame": float(sum(active_counts) / len(active_counts)) if active_counts else 0.0,
        "average_front_free_space_m": (
            float(sum(front_free_space_values) / len(front_free_space_values))
            if front_free_space_values
            else None
        ),
        "maneuver_counts": dict(maneuver_counts),
        "risk_counts": dict(risk_counts),
        "max_route_progress_m": (
            float(max(item.ego.route_progress_m for item in world_states)) if world_states else 0.0
        ),
    }
