from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .constants import MODEL_CAMERA_ORDER, RADAR_SENSOR_ORDER

LOGGER = logging.getLogger(__name__)

BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

RADAR_SENSOR_COLORS = {
    "RADAR_FRONT": "tab:red",
    "RADAR_FRONT_LEFT": "tab:orange",
    "RADAR_FRONT_RIGHT": "tab:pink",
    "RADAR_BACK_LEFT": "tab:green",
    "RADAR_BACK_RIGHT": "tab:blue",
}


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def _lidar_boxes_and_scores(output: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = output.get("boxes_3d")
    if boxes is None or len(boxes) == 0:
        return (
            np.zeros((0, 8, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    corners = boxes.corners.detach().cpu().numpy()
    scores = output["scores_3d"].detach().cpu().numpy()
    labels = output["labels_3d"].detach().cpu().numpy()
    return corners, scores, labels


def _render_prediction_bev(
    *,
    ax,
    lidar_points: np.ndarray,
    radar_points: np.ndarray,
    output: Dict[str, Any],
    class_names: Sequence[str],
    score_threshold: float,
    title: str,
) -> None:
    if len(lidar_points):
        ax.scatter(lidar_points[:, 0], lidar_points[:, 1], s=0.25, c="black", alpha=0.35, label="lidar")
    if len(radar_points):
        ax.scatter(radar_points[:, 0], radar_points[:, 1], s=3.0, c="tab:red", alpha=0.7, label="radar")

    corners, scores, labels = _lidar_boxes_and_scores(output)
    for box_corners, score, label in zip(corners, scores, labels):
        if float(score) < score_threshold:
            continue
        bev = box_corners[:4, :2]
        loop = np.concatenate([bev, bev[:1]], axis=0)
        ax.plot(loop[:, 0], loop[:, 1], linewidth=1.5)
        class_name = class_names[int(label)] if int(label) < len(class_names) else str(int(label))
        ax.text(
            float(bev[:, 0].mean()),
            float(bev[:, 1].mean()),
            f"{class_name} {score:.2f}",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    ax.set_xlabel("x (forward, m)")
    ax.set_ylabel("y (left, m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")


def save_bev_plot(
    *,
    output_path: Path,
    lidar_points: np.ndarray,
    radar_points: np.ndarray,
    output: Dict[str, Any],
    class_names: Sequence[str],
    score_threshold: float = 0.2,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    _render_prediction_bev(
        ax=ax,
        lidar_points=lidar_points,
        radar_points=radar_points,
        output=output,
        class_names=class_names,
        score_threshold=score_threshold,
        title="BEVFusion prediction debug",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _project_points(points_xyz: np.ndarray, lidar2image_aug: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if points_xyz.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([points_xyz, ones], axis=1)
    projected = points_h @ lidar2image_aug.T
    depth = projected[:, 2]
    valid = depth > 1e-4
    projected[valid, 0] /= depth[valid]
    projected[valid, 1] /= depth[valid]
    return projected[:, :2], depth


def save_camera_overlays(
    *,
    output_dir: Path,
    images_for_viz: Sequence[np.ndarray],
    lidar2image: np.ndarray,
    img_aug_matrix: np.ndarray,
    output: Dict[str, Any],
    class_names: Sequence[str],
    score_threshold: float = 0.2,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    corners, scores, labels = _lidar_boxes_and_scores(output)
    for cam_idx, camera_name in enumerate(MODEL_CAMERA_ORDER):
        image = Image.fromarray(images_for_viz[cam_idx])
        draw = ImageDraw.Draw(image)
        lidar2image_aug = img_aug_matrix[cam_idx] @ lidar2image[cam_idx]

        for box_corners, score, label in zip(corners, scores, labels):
            if float(score) < score_threshold:
                continue
            projected, depth = _project_points(box_corners, lidar2image_aug)
            valid = depth > 1e-4
            if valid.sum() < 8:
                continue
            for edge_start, edge_end in BOX_EDGES:
                p0 = projected[edge_start]
                p1 = projected[edge_end]
                draw.line((float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])), fill=(255, 64, 64), width=2)
            label_name = class_names[int(label)] if int(label) < len(class_names) else str(int(label))
            top_left = projected[np.argmin(projected[:, 1])]
            draw.text((float(top_left[0]), float(top_left[1])), f"{label_name} {score:.2f}", fill=(255, 255, 0))

        image.save(output_dir / f"{camera_name}.png")


def save_geometry_topdown(
    *,
    output_path: Path,
    lidar_points: np.ndarray,
    radar_by_sensor: Mapping[str, np.ndarray],
    velocity_by_sensor: Mapping[str, np.ndarray] | None = None,
    color_by: str = "sensor",
    title: str = "Radar / LiDAR geometry sanity",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = None

    if len(lidar_points):
        ax.scatter(lidar_points[:, 0], lidar_points[:, 1], s=0.2, c="black", alpha=0.25, label="lidar")

    velocity_by_sensor = velocity_by_sensor or {}
    for sensor_name in RADAR_SENSOR_ORDER:
        points = radar_by_sensor.get(sensor_name)
        if points is None or len(points) == 0:
            continue
        if color_by == "velocity":
            velocities = velocity_by_sensor.get(sensor_name)
            scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                c=velocities if velocities is not None else np.zeros((points.shape[0],), dtype=np.float32),
                s=10.0,
                cmap="coolwarm",
                alpha=0.85,
                vmin=-10.0,
                vmax=10.0,
                label=sensor_name,
            )
        else:
            scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                s=10.0,
                color=RADAR_SENSOR_COLORS.get(sensor_name, "tab:red"),
                alpha=0.85,
                label=sensor_name,
            )
    if color_by == "velocity" and scatter is not None:
        fig.colorbar(scatter, ax=ax, shrink=0.8, label="radial velocity (m/s)")

    ax.set_xlabel("x (forward, m)")
    ax.set_ylabel("y (left, m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_geometry_sanity_bundle(
    *,
    output_root: str | Path,
    lidar_points: np.ndarray,
    radar_by_sensor: Mapping[str, np.ndarray],
    velocity_by_sensor: Mapping[str, np.ndarray],
    report: Dict[str, Any],
    color_by: str = "sensor",
) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    save_geometry_topdown(
        output_path=output_root / "radar_lidar_topdown.png",
        lidar_points=lidar_points,
        radar_by_sensor=radar_by_sensor,
        velocity_by_sensor=velocity_by_sensor,
        color_by=color_by,
    )
    with (output_root / "geometry_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=_json_default)
    LOGGER.info("Saved geometry sanity bundle to %s", output_root)


def save_prediction_debug_bundle(
    *,
    output_root: str | Path,
    debug: Dict[str, Any],
    output: Dict[str, Any],
    score_threshold: float = 0.2,
) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    class_names = debug.get("object_classes", [])

    save_bev_plot(
        output_path=output_root / "bev_debug.png",
        lidar_points=debug["lidar_points"],
        radar_points=debug["radar_points"],
        output=output,
        class_names=class_names,
        score_threshold=score_threshold,
    )
    save_camera_overlays(
        output_dir=output_root / "camera_overlays",
        images_for_viz=debug["images_for_viz"],
        lidar2image=debug["lidar2image"],
        img_aug_matrix=debug["img_aug_matrix"],
        output=output,
        class_names=class_names,
        score_threshold=score_threshold,
    )
    LOGGER.info("Saved prediction debug bundle to %s", output_root)


def save_prediction_comparison_bundle(
    *,
    output_root: str | Path,
    baseline_debug: Dict[str, Any],
    baseline_output: Dict[str, Any],
    bridge_debug: Dict[str, Any],
    bridge_output: Dict[str, Any],
    score_threshold: float = 0.2,
) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    class_names = bridge_debug.get("object_classes", baseline_debug.get("object_classes", []))
    _render_prediction_bev(
        ax=axes[0],
        lidar_points=baseline_debug["lidar_points"],
        radar_points=baseline_debug["radar_points"],
        output=baseline_output,
        class_names=class_names,
        score_threshold=score_threshold,
        title="Baseline: zero radar BEV",
    )
    _render_prediction_bev(
        ax=axes[1],
        lidar_points=bridge_debug["lidar_points"],
        radar_points=bridge_debug["radar_points"],
        output=bridge_output,
        class_names=class_names,
        score_threshold=score_threshold,
        title="Bridge: minimal radar input",
    )
    fig.tight_layout()
    fig.savefig(output_root / "bev_comparison.png", dpi=160)
    plt.close(fig)
    LOGGER.info("Saved prediction comparison bundle to %s", output_root)
