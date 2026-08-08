from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt


def _save_behavior_timeline(output_path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [record["carla_frame"] for record in records]
    requested = [record["execution_request"]["requested_behavior"] for record in records]
    executed = [record["execution_state"]["current_behavior"] for record in records]
    labels = sorted(set(requested + executed))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].step(frames, [label_to_idx[label] for label in requested], where="mid")
    axes[0].set_ylabel("Requested")
    axes[0].set_yticks(list(label_to_idx.values()))
    axes[0].set_yticklabels(list(label_to_idx.keys()))
    axes[0].grid(True, alpha=0.3)

    axes[1].step(frames, [label_to_idx[label] for label in executed], where="mid", color="tab:orange")
    axes[1].set_ylabel("Executed")
    axes[1].set_yticks(list(label_to_idx.values()))
    axes[1].set_yticklabels(list(label_to_idx.keys()))
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(frames, [record["execution_state"]["speed_mps"] for record in records], label="speed")
    axes[2].plot(frames, [record["execution_request"]["target_speed_mps"] for record in records], label="target")
    axes[2].set_ylabel("Speed mps")
    axes[2].set_xlabel("CARLA frame")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_lane_change_timeline(output_path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [record["carla_frame"] for record in records]
    current_lane = [record["execution_state"]["current_lane_id"] or 0 for record in records]
    target_lane = [record["execution_state"]["target_lane_id"] or 0 for record in records]
    progress = [record["execution_state"]["lane_change_progress"] for record in records]
    lateral_shift = [record["execution_state"]["lateral_shift_m"] for record in records]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].step(frames, current_lane, where="mid", label="current lane")
    axes[0].step(frames, target_lane, where="mid", label="target lane")
    axes[0].set_ylabel("Lane id")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(frames, progress, color="tab:green")
    axes[1].set_ylabel("LC progress")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(frames, lateral_shift, color="tab:red")
    axes[2].set_ylabel("Lateral shift m")
    axes[2].set_xlabel("CARLA frame")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_stage3c_visualization_bundle(
    *,
    output_root: str | Path,
    execution_records: Iterable[Dict[str, Any]],
) -> None:
    records = list(execution_records)
    output_root = Path(output_root)
    _save_behavior_timeline(output_root / "behavior_execution_timeline.png", records)
    _save_lane_change_timeline(output_root / "lane_change_timeline.png", records)
