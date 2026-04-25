"""
carla_bevfusion_stage1/bevfusion_live_adapter.py
=================================================
Converts a LiveFrame (raw CARLA sensor bundle) into BEVFusion model inputs
and runs online inference, returning normalised detections.

Design rules:
  - Calls run_bevfusion_inference() from bevfusion_runtime.py – never reimplements inference.
  - Calls BEVFusionSampleAdapter helpers internally – never reimplements preprocessing.
  - Converts results to a canonical DetectionList for WorldStateBuilder.
  - Thread-safe: build one instance, call adapt_and_infer() from any thread.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .adapter import BEVFusionSampleAdapter, _image_to_normalized_tensor, _range_mask_3d, radar_native_to_cartesian
from .bevfusion_runtime import run_bevfusion_inference
from .carla_sensor_sync import LiveCalibration, LiveFrame
from .constants import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    DEFAULT_POINT_CLOUD_RANGE,
    LIDAR_SENSOR_NAME,
    MINIMAL_RADAR_COLUMNS,
    MODEL_CAMERA_ORDER,
    RADAR_SENSOR_ORDER,
)
from .coordinate_utils import (
    convert_lidar_points_carla_to_bevfusion,
    homogeneous_camera_intrinsics,
    invert_homogeneous,
    make_image_aug_matrix,
)

LOGGER = logging.getLogger(__name__)


# ── Canonical detection output ────────────────────────────────────────────────

@dataclass
class Detection3D:
    """
    One 3-D bounding box in BEVFusion (LiDAR-ego) frame.
    x, y, z   : centre position (metres, ego-centric BEVFusion coords)
    dx, dy, dz: box half-extents
    yaw_rad   : heading angle (rad)
    score     : confidence ∈ [0, 1]
    label_idx : integer class index
    label_name: human-readable class name (if model.CLASSES is populated)
    """
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    yaw_rad: float
    score: float
    label_idx: int
    label_name: str = ""


@dataclass
class DetectionList:
    frame_id: int
    timestamp_s: float
    detections: List[Detection3D] = field(default_factory=list)
    inference_time_ms: float = 0.0
    num_raw_boxes: int = 0


# ── Pre-processing helpers (live path, no file I/O) ───────────────────────────

def _bgra_to_rgb_pil(bgra: np.ndarray) -> Image.Image:
    """(H, W, 4) uint8 BGRA → PIL RGB."""
    rgb = bgra[:, :, :3][:, :, ::-1].copy()   # BGRA → RGB
    return Image.fromarray(rgb, "RGB")


def _build_live_image_tensor(
    images_bgra: Dict[str, np.ndarray],
    final_dim: Tuple[int, int],
    resize: float,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Process 6 BGRA images → stacked float tensor (6, 3, H, W) + img_aug_matrix (6,4,4).
    Returns (image_tensor, aug_matrices_np).
    """
    tensors: List[torch.Tensor] = []
    aug_matrices: List[np.ndarray] = []
    h_orig, w_orig = next(iter(images_bgra.values())).shape[:2]

    for cam_name in MODEL_CAMERA_ORDER:
        pil_img = _bgra_to_rgb_pil(images_bgra[cam_name])
        # Resize
        new_w = int(w_orig * resize)
        new_h = int(h_orig * resize)
        pil_img = pil_img.resize((new_w, new_h))
        # Centre crop
        fh, fw = int(final_dim[0]), int(final_dim[1])
        crop_h = max(0, new_h - fh)
        crop_w = max(0, (new_w - fw) // 2)
        pil_img = pil_img.crop((crop_w, crop_h, crop_w + fw, crop_h + fh))

        tensors.append(_image_to_normalized_tensor(pil_img, mean, std))
        aug_mat = make_image_aug_matrix(w_orig, h_orig, final_dim=(fh, fw), resize=resize)
        aug_matrices.append(aug_mat)

    stacked = torch.stack(tensors, dim=0).float()
    return stacked, np.stack(aug_matrices, axis=0).astype(np.float32)


def _build_live_lidar(
    lidar_xyzi_carla: np.ndarray,
    point_cloud_range: Tuple[float, ...],
    lidar_max_points: int = 0,
) -> np.ndarray:
    """CARLA lidar → BEVFusion 5-D (x,y,z,intensity,time_diff=0), filtered to range."""
    # Safety check for non-finite points
    mask_finite = np.isfinite(lidar_xyzi_carla).all(axis=1)
    if not mask_finite.all():
        lidar_xyzi_carla = lidar_xyzi_carla[mask_finite]

    pts = convert_lidar_points_carla_to_bevfusion(lidar_xyzi_carla)           # (N,4)
    pts5 = np.concatenate([pts[:, :4], np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1)
    if len(pts5):
        pts5 = pts5[_range_mask_3d(pts5[:, :3], point_cloud_range)]
    if lidar_max_points > 0 and pts5.shape[0] > lidar_max_points:
        pts5 = pts5[:lidar_max_points]
    return pts5.astype(np.float32)


def _build_live_radar(
    radar_raw: Dict[str, np.ndarray],
    calibration: LiveCalibration,
    point_cloud_range: Tuple[float, ...],
    radar_max_points: int = 2500,
) -> np.ndarray:
    """
    Merge all radar sensors → BEVFusion minimal 6-D format.
    Columns: [x_lidar, y_lidar, z_lidar, vx_comp, vy_comp, time_diff=0].
    """
    rows: List[np.ndarray] = []
    world_from_lidar = calibration.world_from_lidar_bev
    lidar_from_world = invert_homogeneous(world_from_lidar)

    for sensor_name in RADAR_SENSOR_ORDER:
        raw = radar_raw.get(sensor_name, np.zeros((0, 4), dtype=np.float32))
        
        # Safety check for non-finite points
        mask_finite = np.isfinite(raw).all(axis=1)
        if not mask_finite.all():
            raw = raw[mask_finite]

        xyz_sensor, radial_vel, los_sensor = radar_native_to_cartesian(raw)
        if xyz_sensor.shape[0] == 0:
            rows.append(np.zeros((0, len(MINIMAL_RADAR_COLUMNS)), dtype=np.float32))
            continue

        # Bring xyz to current-lidar frame via world
        from .coordinate_utils import transform_points
        world_from_sensor = calibration.world_from_cameras_bev.get(sensor_name)
        if world_from_sensor is None:
            # Radar not in camera dict – skip
            rows.append(np.zeros((0, len(MINIMAL_RADAR_COLUMNS)), dtype=np.float32))
            continue
        xyz_lidar = transform_points(xyz_sensor, lidar_from_world @ world_from_sensor)

        # BEV range filter
        from .adapter import _range_mask_bev
        mask = _range_mask_bev(xyz_lidar, point_cloud_range)
        xyz_lidar = xyz_lidar[mask]
        vel_xy = (radial_vel[mask].reshape(-1, 1) * los_sensor[mask][:, :2]).astype(np.float32)
        time_col = np.zeros((xyz_lidar.shape[0], 1), dtype=np.float32)
        encoded = np.concatenate([xyz_lidar, vel_xy, time_col], axis=1).astype(np.float32)
        rows.append(encoded)

    merged = np.concatenate(rows, axis=0) if rows else np.zeros((0, len(MINIMAL_RADAR_COLUMNS)), dtype=np.float32)
    if merged.shape[0] > radar_max_points:
        merged = merged[:radar_max_points]
    return merged.astype(np.float32)


def _build_calibration_tensors(
    calibration: LiveCalibration,
    aug_matrices_np: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """Pack calibration matrices into tensors expected by BEVFusion forward()."""
    world_from_lidar = calibration.world_from_lidar_bev
    lidar_from_world = invert_homogeneous(world_from_lidar)

    camera2ego_list: List[np.ndarray] = []
    lidar2camera_list: List[np.ndarray] = []
    lidar2image_list: List[np.ndarray] = []
    camera_intrinsics_list: List[np.ndarray] = []
    camera2lidar_list: List[np.ndarray] = []

    for cam_name in MODEL_CAMERA_ORDER:
        ego_from_cam = calibration.ego_from_cameras_bev[cam_name]
        world_from_cam = calibration.world_from_cameras_bev[cam_name]
        intrinsics_3x3 = calibration.camera_intrinsics[cam_name]
        intrinsics_4x4 = homogeneous_camera_intrinsics(intrinsics_3x3)

        cam2lidar = (lidar_from_world @ world_from_cam).astype(np.float32)
        lidar2cam = invert_homogeneous(cam2lidar)
        lidar2img = (intrinsics_4x4 @ lidar2cam).astype(np.float32)

        camera2ego_list.append(ego_from_cam)
        lidar2camera_list.append(lidar2cam)
        lidar2image_list.append(lidar2img)
        camera_intrinsics_list.append(intrinsics_4x4)
        camera2lidar_list.append(cam2lidar)

    lidar_aug = np.eye(4, dtype=np.float32)
    ego_from_lidar = calibration.ego_from_lidar_bev

    return {
        "camera2ego": torch.from_numpy(np.stack(camera2ego_list, axis=0)).float(),
        "lidar2ego": torch.from_numpy(ego_from_lidar).float(),
        "lidar2camera": torch.from_numpy(np.stack(lidar2camera_list, axis=0)).float(),
        "lidar2image": torch.from_numpy(np.stack(lidar2image_list, axis=0)).float(),
        "camera_intrinsics": torch.from_numpy(np.stack(camera_intrinsics_list, axis=0)).float(),
        "camera2lidar": torch.from_numpy(np.stack(camera2lidar_list, axis=0)).float(),
        "img_aug_matrix": torch.from_numpy(aug_matrices_np).float(),
        "lidar_aug_matrix": torch.from_numpy(lidar_aug).float(),
    }


def _parse_detections(
    model_output: List[Dict[str, Any]],
    frame_id: int,
    timestamp_s: float,
    object_classes: Tuple[str, ...],
    score_threshold: float,
    inference_time_ms: float,
) -> DetectionList:
    """Parse raw BEVFusion output dicts into a DetectionList."""
    result = DetectionList(
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        inference_time_ms=inference_time_ms,
    )
    if not model_output:
        return result

    out = model_output[0]
    boxes = out.get("boxes_3d", None)
    scores = out.get("scores_3d", None)
    labels = out.get("labels_3d", None)

    if boxes is None or scores is None or labels is None:
        return result

    # boxes_3d is a LiDARInstance3DBoxes: .tensor shape (N, 9)
    # columns: [x, y, z, dx, dy, dz, yaw, vx, vy]
    try:
        boxes_np = boxes.tensor.cpu().numpy()
    except Exception:
        boxes_np = np.asarray(boxes).reshape(-1, 9)
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()

    result.num_raw_boxes = int(boxes_np.shape[0])

    for i in range(boxes_np.shape[0]):
        score = float(scores_np[i])
        if score < score_threshold:
            continue
        label_idx = int(labels_np[i])
        label_name = object_classes[label_idx] if label_idx < len(object_classes) else str(label_idx)
        b = boxes_np[i]
        result.detections.append(Detection3D(
            x=float(b[0]), y=float(b[1]), z=float(b[2]),
            dx=float(b[3]), dy=float(b[4]), dz=float(b[5]),
            yaw_rad=float(b[6]),
            score=score,
            label_idx=label_idx,
            label_name=label_name,
        ))
    return result


# ── Main adapter class ────────────────────────────────────────────────────────

class BEVFusionLiveAdapter:
    """
    Online BEVFusion inference adapter for live CARLA frames.

    Usage:
        model, cfg, BoxClass = build_bevfusion_model(...)
        adapter = BEVFusionLiveAdapter(model, cfg, BoxClass, device="cuda")
        det_list = adapter.adapt_and_infer(live_frame)
    """

    def __init__(
        self,
        model,              # output of build_bevfusion_model()
        cfg,                # mmcv.Config
        box_type_3d_cls,    # LiDARInstance3DBoxes
        *,
        device: str = "cuda",
        score_threshold: float = 0.35,
        lidar_max_points: int = 0,
        radar_max_points: int = 2500,
    ) -> None:
        self._model = model
        self._cfg = cfg
        self._box_cls = box_type_3d_cls
        self._device = device
        self._score_threshold = score_threshold
        self._lidar_max_points = lidar_max_points
        self._radar_max_points = radar_max_points

        # Pull preprocessing params from cfg (same as BEVFusionSampleAdapter)
        from .adapter import BEVFusionSampleAdapter
        from .bevfusion_runtime import load_runtime_config
        resolved_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg._cfg_dict)

        _tmp = BEVFusionSampleAdapter(resolved_cfg)
        self._final_dim: Tuple[int, int] = tuple(_tmp.final_dim)       # type: ignore
        self._resize: float = float(_tmp.resize_lim[0])
        self._mean: Tuple[float, ...] = _tmp.mean
        self._std: Tuple[float, ...] = _tmp.std
        self._point_cloud_range: Tuple[float, ...] = _tmp.point_cloud_range
        self._object_classes: Tuple[str, ...] = tuple(
            getattr(model, "CLASSES", _tmp.object_classes)
        )
        LOGGER.info(
            "BEVFusionLiveAdapter ready | device=%s | classes=%s",
            device, list(self._object_classes)
        )

    # ── Hot path ──────────────────────────────────────────────────────────────

    def adapt_and_infer(self, frame: LiveFrame) -> DetectionList:
        """
        Convert a LiveFrame → BEVFusion model inputs → run inference →
        return DetectionList.
        Called every tick (10 Hz target on RTX 5060 Ti 16 GB).
        """
        t0 = time.monotonic()

        # 1. Images → tensor
        img_tensor, aug_matrices = _build_live_image_tensor(
            frame.images_bgra,
            final_dim=self._final_dim,
            resize=self._resize,
            mean=self._mean,
            std=self._std,
        )

        # 2. Lidar → 5-D tensor
        lidar_np = _build_live_lidar(
            frame.lidar_xyzi,
            self._point_cloud_range,
            self._lidar_max_points,
        )

        # 3. Radar → 6-D tensor (optional, zeros if no radar)
        radar_np = _build_live_radar(
            frame.radar_raw,
            frame.calibration,
            self._point_cloud_range,
            self._radar_max_points,
        )

        # Safety Check: CUDA `hard_voxelize` will literally crash with `invalid configuration argument` 
        # if the input tensor has 0 points (it causes grid dimension calculations to yield 0).
        # We must inject at least 1 dummy neutral point if the array is empty.
        if lidar_np.shape[0] == 0:
            dummy_lidar = np.zeros((1, 5), dtype=np.float32)
            lidar_np = np.concatenate([lidar_np, dummy_lidar], axis=0)
            
        if radar_np.shape[0] == 0:
            dummy_radar = np.zeros((1, 6), dtype=np.float32)
            radar_np = np.concatenate([radar_np, dummy_radar], axis=0)

        # 4. Calibration tensors
        calib = _build_calibration_tensors(frame.calibration, aug_matrices)

        # 5. Assemble model input batch
        model_inputs: Dict[str, Any] = {
            "img": img_tensor.unsqueeze(0).to(self._device),
            "points": [torch.from_numpy(lidar_np).to(self._device)],
            "radar": [torch.from_numpy(radar_np).to(self._device)],
            **{k: v.unsqueeze(0).to(self._device) for k, v in calib.items()},
            "metas": [{
                "token": f"live_{frame.frame_id}",
                "sample_idx": f"live_{frame.frame_id}",
                "box_type_3d": self._box_cls,
            }],
        }

        # 6. Inference (uses existing run_bevfusion_inference, no_grad inside)
        model_output = run_bevfusion_inference(self._model, model_inputs)
        inference_ms = (time.monotonic() - t0) * 1000.0

        LOGGER.debug(
            "BEVFusion inference frame=%d  %.1f ms  raw_boxes=%s",
            frame.frame_id, inference_ms,
            model_output[0].get("boxes_3d", "?") if model_output else "?",
        )

        return _parse_detections(
            model_output=model_output,
            frame_id=frame.frame_id,
            timestamp_s=frame.timestamp_s,
            object_classes=self._object_classes,
            score_threshold=self._score_threshold,
            inference_time_ms=inference_ms,
        )
