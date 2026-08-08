from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt


def _save_behavior_timeline(output_path: Path, frame_records: List[Dict[str, Any]]) -> None:
    if not frame_records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [record["frame_id"] for record in frame_records]
    stage3a_names = [record["stage3a_behavior_request"]["requested_behavior"] for record in frame_records]
    stage3b_names = [record["behavior_request_v2"]["requested_behavior"] for record in frame_records]
    route_options = [record["route_context"]["route_option"] for record in frame_records]
    all_labels = sorted(set(stage3a_names + stage3b_names + route_options))
    label_to_id = {label: index for index, label in enumerate(all_labels)}

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].step(frames, [label_to_id[name] for name in stage3a_names], where="mid")
    axes[0].set_ylabel("Stage3A")
    axes[0].set_yticks(list(label_to_id.values()))
    axes[0].set_yticklabels(list(label_to_id.keys()))
    axes[0].grid(True, alpha=0.3)

    axes[1].step(frames, [label_to_id[name] for name in stage3b_names], where="mid", color="tab:orange")
    axes[1].set_ylabel("Stage3B")
    axes[1].set_yticks(list(label_to_id.values()))
    axes[1].set_yticklabels(list(label_to_id.keys()))
    axes[1].grid(True, alpha=0.3)

    axes[2].step(frames, [label_to_id[name] for name in route_options], where="mid", color="tab:green")
    axes[2].set_ylabel("Route option")
    axes[2].set_xlabel("Frame")
    axes[2].set_yticks(list(label_to_id.values()))
    axes[2].set_yticklabels(list(label_to_id.keys()))
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_lane_change_stage_timeline(output_path: Path, frame_records: List[Dict[str, Any]]) -> None:
    if not frame_records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [record["frame_id"] for record in frame_records]
    stage_names = [record["behavior_selection"]["lane_change_stage"] for record in frame_records]
    route_influence = [1.0 if record["behavior_selection"]["route_influence"] else 0.0 for record in frame_records]
    target_speeds = [record["behavior_request_v2"]["target_speed_mps"] for record in frame_records]
    labels = ["none", "prepare", "commit"]
    label_to_id = {label: index for index, label in enumerate(labels)}

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].step(frames, [label_to_id.get(name, 0) for name in stage_names], where="mid", color="tab:red")
    axes[0].set_ylabel("LC stage")
    axes[0].set_yticks(list(label_to_id.values()))
    axes[0].set_yticklabels(list(label_to_id.keys()))
    axes[0].grid(True, alpha=0.3)

    axes[1].step(frames, route_influence, where="mid", color="tab:purple")
    axes[1].set_ylabel("Route influence")
    axes[1].set_yticks([0, 1])
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(frames, target_speeds, marker="o", color="tab:blue")
    axes[2].set_ylabel("Target speed")
    axes[2].set_xlabel("Frame")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_stage3b_visualization_bundle(
    *,
    output_root: str | Path,
    frame_records: Iterable[Dict[str, Any]],
) -> None:
    frame_records = list(frame_records)
    output_root = Path(output_root)
    _save_behavior_timeline(output_root / "behavior_timeline_v2.png", frame_records)
    _save_lane_change_stage_timeline(output_root / "lane_change_stage_timeline.png", frame_records)
