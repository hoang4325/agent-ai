from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt


def _save_behavior_timeline(output_path: Path, frame_records: List[Dict[str, Any]]) -> None:
    if not frame_records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [record["frame_id"] for record in frame_records]
    stage2_names = [record["stage2_decision"]["maneuver"] for record in frame_records]
    stage3_names = [record["behavior_request"]["requested_behavior"] for record in frame_records]
    target_speeds = [record["behavior_request"]["target_speed_mps"] for record in frame_records]
    all_labels = sorted(set(stage2_names + stage3_names))
    label_to_id = {label: index for index, label in enumerate(all_labels)}

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].step(frames, [label_to_id[name] for name in stage2_names], where="mid")
    axes[0].set_ylabel("Stage2")
    axes[0].set_yticks(list(label_to_id.values()))
    axes[0].set_yticklabels(list(label_to_id.keys()))
    axes[0].grid(True, alpha=0.3)

    axes[1].step(frames, [label_to_id[name] for name in stage3_names], where="mid", color="tab:orange")
    axes[1].set_ylabel("Stage3A")
    axes[1].set_yticks(list(label_to_id.values()))
    axes[1].set_yticklabels(list(label_to_id.keys()))
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(frames, target_speeds, marker="o", color="tab:green")
    axes[2].set_ylabel("Target speed")
    axes[2].set_xlabel("Frame")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_lane_context_timeline(output_path: Path, frame_records: List[Dict[str, Any]]) -> None:
    if not frame_records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [record["frame_id"] for record in frame_records]
    blocked = [1.0 if record["lane_scene"]["current_lane_blocked"] else 0.0 for record in frame_records]
    junction_distance = [
        (
            0.0
            if record["lane_context"]["junction_context"]["is_in_junction"]
            else (
                float(record["lane_context"]["junction_context"]["distance_to_junction_m"])
                if record["lane_context"]["junction_context"]["distance_to_junction_m"] is not None
                else float("nan")
            )
        )
        for record in frame_records
    ]
    lane_change_left = [
        1.0 if record["maneuver_validation"]["lane_change_permission"]["left"] else 0.0
        for record in frame_records
    ]
    lane_change_right = [
        1.0 if record["maneuver_validation"]["lane_change_permission"]["right"] else 0.0
        for record in frame_records
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].step(frames, blocked, where="mid", color="tab:red")
    axes[0].set_ylabel("Current lane blocked")
    axes[0].set_yticks([0, 1])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames, junction_distance, marker="o", color="tab:purple")
    axes[1].set_ylabel("Junction dist (m)")
    axes[1].grid(True, alpha=0.3)

    axes[2].step(frames, lane_change_left, where="mid", label="Left allowed")
    axes[2].step(frames, lane_change_right, where="mid", label="Right allowed")
    axes[2].set_ylabel("Lane change")
    axes[2].set_xlabel("Frame")
    axes[2].set_yticks([0, 1])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_stage3_visualization_bundle(
    *,
    output_root: str | Path,
    frame_records: Iterable[Dict[str, Any]],
) -> None:
    frame_records = list(frame_records)
    output_root = Path(output_root)
    _save_behavior_timeline(output_root / "behavior_timeline.png", frame_records)
    _save_lane_context_timeline(output_root / "lane_context_timeline.png", frame_records)
