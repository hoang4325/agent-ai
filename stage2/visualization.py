from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt

from .object_schema import DecisionIntent, WorldState


def _save_decision_timeline(
    output_path: Path,
    decisions: List[DecisionIntent],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not decisions:
        return

    frames = [item.frame_id for item in decisions]
    speeds = [item.target_speed_mps for item in decisions]
    maneuvers = [item.maneuver for item in decisions]
    unique_maneuvers = {name: index for index, name in enumerate(sorted(set(maneuvers)))}
    maneuver_ids = [unique_maneuvers[name] for name in maneuvers]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(frames, speeds, marker="o")
    axes[0].set_ylabel("Target speed (m/s)")
    axes[0].grid(True, alpha=0.3)

    axes[1].step(frames, maneuver_ids, where="mid")
    axes[1].set_yticks(list(unique_maneuvers.values()))
    axes[1].set_yticklabels(list(unique_maneuvers.keys()))
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Maneuver")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_scene_timeline(
    output_path: Path,
    world_states: List[WorldState],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not world_states:
        return

    frames = [item.frame_id for item in world_states]
    front_free_space = [
        item.scene.front_free_space_m if item.scene.front_free_space_m is not None else float("nan")
        for item in world_states
    ]
    highest_risk = [
        {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}[item.risk_summary.highest_risk_level]
        for item in world_states
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(frames, front_free_space, marker="o")
    axes[0].set_ylabel("Front free space (m)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(frames, highest_risk, marker="o", color="tab:red")
    axes[1].set_ylabel("Highest risk")
    axes[1].set_xlabel("Frame")
    axes[1].set_yticks([0.25, 0.5, 0.75, 1.0])
    axes[1].set_yticklabels(["low", "medium", "high", "critical"])
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_stage2_visualization_bundle(
    *,
    output_root: str | Path,
    world_states: Iterable[WorldState],
    decisions: Iterable[DecisionIntent],
) -> None:
    output_root = Path(output_root)
    world_states = list(world_states)
    decisions = list(decisions)
    _save_decision_timeline(output_root / "decision_timeline.png", decisions)
    _save_scene_timeline(output_root / "scene_timeline.png", world_states)
