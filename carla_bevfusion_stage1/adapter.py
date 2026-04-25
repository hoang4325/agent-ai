from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .config_loader import find_pipeline_step
from .constants import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IMAGE_STD,
    DEFAULT_POINT_CLOUD_RANGE,
    LIDAR_SENSOR_NAME,
    MINIMAL_RADAR_COLUMNS,
    MODEL_CAMERA_ORDER,
    RADAR_SENSOR_ORDER,
    RADAR_TRACE_SELECTED_COLUMNS,
)
from .coordinate_utils import (
    convert_lidar_points_carla_to_bevfusion,
    homogeneous_camera_intrinsics,
    invert_homogeneous,
    transform_points,
)

LOGGER = logging.getLogger(__name__)

SAMPLE_NAME_PATTERN = re.compile(r"sample_(\d+)$")
EPSILON = 1e-6


@dataclass
class AdaptedSample:
    model_inputs: Dict[str, Any]
    debug: Dict[str, Any]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sample_meta(sample_dir: str | Path) -> Dict[str, Any]:
    return _load_json(Path(sample_dir) / "meta.json")


def _sample_sequence_key(sample_dir: Path) -> int:
    match = SAMPLE_NAME_PATTERN.match(sample_dir.name)
    if match:
        return int(match.group(1))
    meta_path = sample_dir / "meta.json"
    if meta_path.exists():
        return int(_load_json(meta_path).get("sequence_index", -1))
    return -1


def _image_to_normalized_tensor(image: Image.Image, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def _range_mask_3d(points_xyz: np.ndarray, point_cloud_range: Sequence[float]) -> np.ndarray:
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    return (
        (points_xyz[:, 0] > x_min)
        & (points_xyz[:, 1] > y_min)
        & (points_xyz[:, 2] > z_min)
        & (points_xyz[:, 0] < x_max)
        & (points_xyz[:, 1] < y_max)
        & (points_xyz[:, 2] < z_max)
    )


def _range_mask_bev(points_xyz: np.ndarray, point_cloud_range: Sequence[float]) -> np.ndarray:
    x_min, y_min, _z_min, x_max, y_max, _z_max = point_cloud_range
    return (
        (points_xyz[:, 0] > x_min)
        & (points_xyz[:, 1] > y_min)
        & (points_xyz[:, 0] < x_max)
        & (points_xyz[:, 1] < y_max)
    )


def _resize_crop_for_test(
    image: Image.Image,
    final_dim: Sequence[int],
    resize_lim: Sequence[float],
    bot_pct_lim: Sequence[float],
) -> Tuple[Image.Image, np.ndarray]:
    original_width, original_height = image.size
    resize = float(sum(resize_lim) / len(resize_lim))
    resized_width = int(original_width * resize)
    resized_height = int(original_height * resize)
    crop_h = int((1.0 - float(sum(bot_pct_lim) / len(bot_pct_lim))) * resized_height) - int(final_dim[0])
    crop_w = int(max(0, resized_width - int(final_dim[1])) / 2)
    crop_box = (
        crop_w,
        crop_h,
        crop_w + int(final_dim[1]),
        crop_h + int(final_dim[0]),
    )
    resized = image.resize((resized_width, resized_height))
    cropped = resized.crop(crop_box)
    img_aug = np.eye(4, dtype=np.float32)
    img_aug[0, 0] = resize
    img_aug[1, 1] = resize
    img_aug[0, 3] = -float(crop_w)
    img_aug[1, 3] = -float(crop_h)
    return cropped, img_aug


def _history_dirs(sample_dir: Path, num_previous: int) -> List[Path]:
    if num_previous <= 0:
        return []
    candidates = [item for item in sample_dir.parent.iterdir() if item.is_dir() and item.name.startswith("sample_")]
    ordered = sorted(candidates, key=_sample_sequence_key)
    current_idx = ordered.index(sample_dir)
    start = max(0, current_idx - num_previous)
    return list(reversed(ordered[start:current_idx]))


def _matrix(meta: Dict[str, Any], sensor_name: str, key: str) -> np.ndarray:
    return np.asarray(meta["sensors"][sensor_name][key], dtype=np.float32)


def _current_lidar_from_sensor(meta_current: Dict[str, Any], meta_other: Dict[str, Any], sensor_name: str) -> np.ndarray:
    world_from_current_lidar = _matrix(
        meta_current, LIDAR_SENSOR_NAME, "matrix_world_from_sensor_bevfusion"
    )
    world_from_other_sensor = _matrix(meta_other, sensor_name, "matrix_world_from_sensor_bevfusion")
    return (invert_homogeneous(world_from_current_lidar) @ world_from_other_sensor).astype(np.float32)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.reshape(0, 3).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.maximum(norms, EPSILON)
    return (vectors / safe).astype(np.float32)


def radar_native_to_cartesian(raw_radar: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert CARLA native radar detections to BEVFusion-handed sensor-frame xyz.

    Returns:
        xyz_sensor_bevfusion: [N, 3]
        radial_velocity: [N]
        los_unit_sensor_bevfusion: [N, 3]
    """
    if raw_radar.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    radial_velocity = raw_radar[:, 0].astype(np.float32)
    altitude = raw_radar[:, 1].astype(np.float32)
    azimuth = raw_radar[:, 2].astype(np.float32)
    depth = raw_radar[:, 3].astype(np.float32)

    cos_alt = np.cos(altitude)
    sin_alt = np.sin(altitude)
    cos_az = np.cos(azimuth)
    sin_az = np.sin(azimuth)

    x = depth * cos_alt * cos_az
    y = depth * cos_alt * sin_az
    z = depth * sin_alt

    xyz_sensor_bevfusion = np.stack([x, -y, z], axis=1).astype(np.float32)
    los_unit_sensor_bevfusion = _normalize_vectors(xyz_sensor_bevfusion)
    return xyz_sensor_bevfusion, radial_velocity, los_unit_sensor_bevfusion


class BEVFusionSampleAdapter:
    def __init__(self, resolved_cfg: Dict[str, Any]) -> None:
        self.cfg = resolved_cfg
        image_aug_cfg = find_pipeline_step(self.cfg, "ImageAug3D")
        image_norm_cfg = find_pipeline_step(self.cfg, "ImageNormalize")

        self.final_dim = tuple(image_aug_cfg.get("final_dim", DEFAULT_IMAGE_SIZE))
        self.resize_lim = tuple(image_aug_cfg.get("resize_lim", (0.48, 0.48)))
        self.bot_pct_lim = tuple(image_aug_cfg.get("bot_pct_lim", (0.0, 0.0)))
        self.mean = tuple(image_norm_cfg.get("mean", DEFAULT_IMAGE_MEAN))
        self.std = tuple(image_norm_cfg.get("std", DEFAULT_IMAGE_STD))
        self.point_cloud_range = tuple(self.cfg.get("point_cloud_range", DEFAULT_POINT_CLOUD_RANGE))
        self.radar_sweeps = int(self.cfg.get("radar_sweeps", 6))
        self.radar_max_points = int(self.cfg.get("radar_max_points", 2500))
        self.lidar_sweeps_test = int(self.cfg.get("lidar_sweeps_test", 9))
        self.lidar_max_points = int(self.cfg.get("lidar_max_points", 0))
        self.radar_compensate_velocity = bool(self.cfg.get("radar_compensate_velocity", True))
        self.object_classes = tuple(self.cfg.get("object_classes", ()))

    def build_from_sample_dir(
        self,
        sample_dir: str | Path,
        *,
        radar_mode: str = "minimal",
        box_type_3d_cls: Any | None = None,
        device: str | torch.device = "cpu",
    ) -> AdaptedSample:
        if radar_mode != "minimal":
            raise ValueError(f"Unsupported radar_mode={radar_mode}. Only 'minimal' is implemented.")

        sample_dir = Path(sample_dir)
        meta_current = _load_json(sample_dir / "meta.json")
        lidar_history = _history_dirs(sample_dir, self.lidar_sweeps_test)
        radar_history = _history_dirs(sample_dir, self.radar_sweeps - 1)

        images_t, images_viz, img_aug_matrices = self._prepare_images(sample_dir)
        lidar_points, lidar_debug = self._prepare_lidar(sample_dir, meta_current, lidar_history)
        radar_points, radar_debug = self._prepare_radar_minimal(sample_dir, meta_current, radar_history)
        calibration = self._prepare_calibration(meta_current, img_aug_matrices)

        batch = {
            "img": images_t.unsqueeze(0).to(device),
            "points": [torch.from_numpy(lidar_points).to(device)],
            "radar": [torch.from_numpy(radar_points).to(device)],
            "camera2ego": calibration["camera2ego"].unsqueeze(0).to(device),
            "lidar2ego": calibration["lidar2ego"].unsqueeze(0).to(device),
            "lidar2camera": calibration["lidar2camera"].unsqueeze(0).to(device),
            "lidar2image": calibration["lidar2image"].unsqueeze(0).to(device),
            "camera_intrinsics": calibration["camera_intrinsics"].unsqueeze(0).to(device),
            "camera2lidar": calibration["camera2lidar"].unsqueeze(0).to(device),
            "img_aug_matrix": calibration["img_aug_matrix"].unsqueeze(0).to(device),
            "lidar_aug_matrix": calibration["lidar_aug_matrix"].unsqueeze(0).to(device),
            "metas": [
                {
                    "token": meta_current["sample_name"],
                    "sample_idx": meta_current["sample_name"],
                    "box_type_3d": box_type_3d_cls,
                }
            ],
        }

        debug = {
            "sample_dir": str(sample_dir),
            "sample_name": meta_current["sample_name"],
            "images_for_viz": images_viz,
            "image_tensor_shape": list(images_t.shape),
            "img_aug_matrix": calibration["img_aug_matrix"].numpy(),
            "lidar2image": calibration["lidar2image"].numpy(),
            "lidar_points": lidar_points,
            "radar_points": radar_points[:, :3],
            "radar_columns": list(MINIMAL_RADAR_COLUMNS),
            "lidar_debug": lidar_debug,
            "radar_debug": radar_debug,
            "meta": meta_current,
            "object_classes": list(self.object_classes),
        }
        return AdaptedSample(model_inputs=batch, debug=debug)

    def _prepare_images(self, sample_dir: Path) -> Tuple[torch.Tensor, List[np.ndarray], np.ndarray]:
        image_tensors: List[torch.Tensor] = []
        viz_images: List[np.ndarray] = []
        img_aug_matrices: List[np.ndarray] = []

        for camera_name in MODEL_CAMERA_ORDER:
            image_path = sample_dir / "images" / f"{camera_name}.png"
            if not image_path.exists():
                raise FileNotFoundError(f"Missing camera image for {camera_name}: {image_path}")
            image = Image.open(image_path).convert("RGB")
            processed, img_aug = _resize_crop_for_test(
                image=image,
                final_dim=self.final_dim,
                resize_lim=self.resize_lim,
                bot_pct_lim=self.bot_pct_lim,
            )
            image_tensors.append(_image_to_normalized_tensor(processed, self.mean, self.std))
            viz_images.append(np.asarray(processed, dtype=np.uint8).copy())
            img_aug_matrices.append(img_aug)

        stacked = torch.stack(image_tensors, dim=0).float()
        return stacked, viz_images, np.stack(img_aug_matrices, axis=0).astype(np.float32)

    def _prepare_lidar(
        self,
        sample_dir: Path,
        meta_current: Dict[str, Any],
        history_dirs: Sequence[Path],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        current_timestamp = float(meta_current["timestamp"])
        current_raw = np.load(sample_dir / "lidar" / f"{LIDAR_SENSOR_NAME}.npy")
        current_points = convert_lidar_points_carla_to_bevfusion(current_raw)
        current_points_5d = np.concatenate(
            [current_points[:, :4], np.zeros((current_points.shape[0], 1), dtype=np.float32)], axis=1
        )
        all_points = [current_points_5d]
        history_report: List[Dict[str, Any]] = []

        for history_dir in history_dirs:
            meta_history = _load_json(history_dir / "meta.json")
            raw = np.load(history_dir / "lidar" / f"{LIDAR_SENSOR_NAME}.npy")
            points = convert_lidar_points_carla_to_bevfusion(raw)
            current_lidar_from_past_lidar = _current_lidar_from_sensor(
                meta_current, meta_history, LIDAR_SENSOR_NAME
            )
            transformed_xyz = transform_points(points[:, :3], current_lidar_from_past_lidar)
            transformed = np.concatenate([transformed_xyz, points[:, 3:4]], axis=1)
            time_diff = np.full(
                (transformed.shape[0], 1),
                fill_value=max(0.0, current_timestamp - float(meta_history["timestamp"])),
                dtype=np.float32,
            )
            transformed = np.concatenate([transformed, time_diff], axis=1)
            all_points.append(transformed.astype(np.float32))
            history_report.append(
                {
                    "sample_name": meta_history["sample_name"],
                    "point_count": int(transformed.shape[0]),
                    "time_diff": float(time_diff[0, 0]) if len(time_diff) else 0.0,
                }
            )

        merged = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 5), dtype=np.float32)
        prefilter_count = int(merged.shape[0])
        if len(merged):
            merged = merged[_range_mask_3d(merged[:, :3], self.point_cloud_range)]
        final_count_before_cap = int(merged.shape[0])
        if self.lidar_max_points > 0 and merged.shape[0] > self.lidar_max_points:
            merged = merged[: self.lidar_max_points, :]
        return merged.astype(np.float32), {
            "keyframe_points": int(current_points_5d.shape[0]),
            "prefilter_point_count": prefilter_count,
            "history_used": history_report,
            "final_point_count_before_cap": final_count_before_cap,
            "final_point_count": int(merged.shape[0]),
            "lidar_max_points": int(self.lidar_max_points),
            "point_cloud_range": list(self.point_cloud_range),
        }

    def _prepare_radar_minimal(
        self,
        sample_dir: Path,
        meta_current: Dict[str, Any],
        history_dirs: Sequence[Path],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        source_dirs = [sample_dir, *history_dirs[: self.radar_sweeps - 1]]
        source_metas = [_load_json(path / "meta.json") for path in source_dirs]
        current_timestamp = float(meta_current["timestamp"])

        rows: List[np.ndarray] = []
        history_report: List[Dict[str, Any]] = []
        raw_total = 0
        in_bev_xy_total = 0
        in_full_xyz_total = 0

        for sensor_name in RADAR_SENSOR_ORDER:
            for sweep_index, (source_dir, source_meta) in enumerate(zip(source_dirs, source_metas)):
                raw = np.load(source_dir / "radar" / f"{sensor_name}.npy")
                xyz_sensor, radial_velocity, los_unit_sensor = radar_native_to_cartesian(raw)
                current_lidar_from_sensor = _current_lidar_from_sensor(meta_current, source_meta, sensor_name)
                rotation = current_lidar_from_sensor[:3, :3]
                xyz_current = transform_points(xyz_sensor, current_lidar_from_sensor)
                los_unit_current = _normalize_vectors(los_unit_sensor @ rotation.T)

                neighbor_meta: Optional[Dict[str, Any]]
                if sweep_index == 0:
                    neighbor_meta = source_metas[1] if len(source_metas) > 1 else None
                else:
                    neighbor_meta = source_metas[sweep_index - 1]

                sensor_velocity_current = self._estimate_sensor_velocity_current_lidar(
                    meta_current=meta_current,
                    source_meta=source_meta,
                    sensor_name=sensor_name,
                    neighbor_meta=neighbor_meta,
                )
                compensated_velocity_xy = self._estimate_compensated_los_velocity(
                    radial_velocity=radial_velocity,
                    los_unit_current=los_unit_current,
                    sensor_velocity_current=sensor_velocity_current,
                )

                time_diff_scalar = max(0.0, current_timestamp - float(source_meta["timestamp"]))
                if self.radar_compensate_velocity and len(xyz_current):
                    xyz_current = xyz_current.copy()
                    xyz_current[:, 0:2] += compensated_velocity_xy * time_diff_scalar

                raw_count = int(xyz_current.shape[0])
                raw_total += raw_count
                bev_mask = _range_mask_bev(xyz_current, self.point_cloud_range) if raw_count else np.zeros((0,), dtype=bool)
                xyz_mask = _range_mask_3d(xyz_current, self.point_cloud_range) if raw_count else np.zeros((0,), dtype=bool)
                in_bev_xy_total += int(bev_mask.sum())
                in_full_xyz_total += int(xyz_mask.sum())

                if raw_count:
                    selected_xyz = xyz_current[bev_mask]
                    selected_velocity_xy = compensated_velocity_xy[bev_mask]
                    selected_time = np.full((selected_xyz.shape[0], 1), time_diff_scalar, dtype=np.float32)
                    encoded = np.concatenate([selected_xyz, selected_velocity_xy, selected_time], axis=1).astype(np.float32)
                else:
                    encoded = np.zeros((0, len(MINIMAL_RADAR_COLUMNS)), dtype=np.float32)

                rows.append(encoded)
                history_report.append(
                    {
                        "sensor": sensor_name,
                        "sample_name": source_meta["sample_name"],
                        "sweep_index": sweep_index,
                        "raw_count": raw_count,
                        "in_bev_xy_count": int(bev_mask.sum()),
                        "in_full_xyz_count": int(xyz_mask.sum()),
                        "time_diff": float(time_diff_scalar),
                        "sensor_velocity_current_lidar": sensor_velocity_current.astype(float).tolist(),
                        "radial_velocity_min": float(radial_velocity.min()) if len(radial_velocity) else 0.0,
                        "radial_velocity_max": float(radial_velocity.max()) if len(radial_velocity) else 0.0,
                    }
                )

        merged = np.concatenate(rows, axis=0) if rows else np.zeros((0, len(MINIMAL_RADAR_COLUMNS)), dtype=np.float32)
        final_count_before_cap = int(merged.shape[0])
        if merged.shape[0] > self.radar_max_points:
            merged = merged[: self.radar_max_points, :]

        report = {
            "bridge_strategy": "minimal_6d_runtime_patch",
            "input_columns": list(MINIMAL_RADAR_COLUMNS),
            "repo_expected_selected_columns": list(RADAR_TRACE_SELECTED_COLUMNS),
            "sensor_velocity_estimation": "finite_difference_in_current_lidar_frame_with_ego_velocity_fallback",
            "history_used": history_report,
            "raw_total": raw_total,
            "in_bev_xy_total": in_bev_xy_total,
            "in_full_xyz_total": in_full_xyz_total,
            "z_out_of_range_total": max(0, in_bev_xy_total - in_full_xyz_total),
            "final_point_count_before_cap": final_count_before_cap,
            "final_point_count": int(merged.shape[0]),
            "point_cloud_range": list(self.point_cloud_range),
            "radar_sweeps_requested": self.radar_sweeps,
            "radar_sweeps_used": len(source_dirs),
        }
        return merged.astype(np.float32), report

    def _estimate_sensor_velocity_current_lidar(
        self,
        *,
        meta_current: Dict[str, Any],
        source_meta: Dict[str, Any],
        sensor_name: str,
        neighbor_meta: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        if neighbor_meta is not None:
            dt = float(neighbor_meta["timestamp"]) - float(source_meta["timestamp"])
            if abs(dt) > EPSILON:
                source_pose = _current_lidar_from_sensor(meta_current, source_meta, sensor_name)
                neighbor_pose = _current_lidar_from_sensor(meta_current, neighbor_meta, sensor_name)
                velocity = (neighbor_pose[:3, 3] - source_pose[:3, 3]) / dt
                return velocity.astype(np.float32)

        world_from_current_lidar = _matrix(meta_current, LIDAR_SENSOR_NAME, "matrix_world_from_sensor_bevfusion")
        world_from_source_ego = np.asarray(
            source_meta["ego"]["matrix_world_from_ego_bevfusion"], dtype=np.float32
        )
        current_lidar_from_source_ego = (
            invert_homogeneous(world_from_current_lidar) @ world_from_source_ego
        ).astype(np.float32)
        ego_velocity = np.asarray(source_meta["ego"]["velocity_carla"], dtype=np.float32)
        ego_velocity[1] *= -1.0
        return (ego_velocity @ current_lidar_from_source_ego[:3, :3].T).astype(np.float32)

    def _estimate_compensated_los_velocity(
        self,
        *,
        radial_velocity: np.ndarray,
        los_unit_current: np.ndarray,
        sensor_velocity_current: np.ndarray,
    ) -> np.ndarray:
        if len(radial_velocity) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        sensor_radial = los_unit_current @ sensor_velocity_current.reshape(3, 1)
        compensated_radial = radial_velocity.reshape(-1, 1) + sensor_radial
        compensated_xy = compensated_radial * los_unit_current[:, :2]
        return compensated_xy.astype(np.float32)

    def _prepare_calibration(
        self,
        meta_current: Dict[str, Any],
        img_aug_matrices: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        world_from_lidar = _matrix(meta_current, LIDAR_SENSOR_NAME, "matrix_world_from_sensor_bevfusion")
        ego_from_lidar = np.asarray(
            meta_current["sensors"][LIDAR_SENSOR_NAME]["matrix_ego_from_sensor_bevfusion"],
            dtype=np.float32,
        )

        camera2ego_list: List[np.ndarray] = []
        lidar2camera_list: List[np.ndarray] = []
        lidar2image_list: List[np.ndarray] = []
        camera_intrinsics_list: List[np.ndarray] = []
        camera2lidar_list: List[np.ndarray] = []

        for camera_name in MODEL_CAMERA_ORDER:
            camera2ego = np.asarray(
                meta_current["sensors"][camera_name]["matrix_ego_from_sensor_bevfusion"],
                dtype=np.float32,
            )
            camera_intrinsics = homogeneous_camera_intrinsics(
                np.asarray(meta_current["sensors"][camera_name]["camera_intrinsics"], dtype=np.float32)
            )
            world_from_camera = _matrix(meta_current, camera_name, "matrix_world_from_sensor_bevfusion")
            camera2lidar = (invert_homogeneous(world_from_lidar) @ world_from_camera).astype(np.float32)
            lidar2camera = invert_homogeneous(camera2lidar)
            lidar2image = (camera_intrinsics @ lidar2camera).astype(np.float32)

            camera2ego_list.append(camera2ego)
            lidar2camera_list.append(lidar2camera)
            lidar2image_list.append(lidar2image)
            camera_intrinsics_list.append(camera_intrinsics)
            camera2lidar_list.append(camera2lidar)

        lidar_aug_matrix = np.eye(4, dtype=np.float32)
        return {
            "camera2ego": torch.from_numpy(np.stack(camera2ego_list, axis=0)).float(),
            "lidar2ego": torch.from_numpy(ego_from_lidar).float(),
            "lidar2camera": torch.from_numpy(np.stack(lidar2camera_list, axis=0)).float(),
            "lidar2image": torch.from_numpy(np.stack(lidar2image_list, axis=0)).float(),
            "camera_intrinsics": torch.from_numpy(np.stack(camera_intrinsics_list, axis=0)).float(),
            "camera2lidar": torch.from_numpy(np.stack(camera2lidar_list, axis=0)).float(),
            "img_aug_matrix": torch.from_numpy(img_aug_matrices).float(),
            "lidar_aug_matrix": torch.from_numpy(lidar_aug_matrix).float(),
        }
