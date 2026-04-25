from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .constants import DEFAULT_IMAGE_SIZE

CARLA_TO_BEVFUSION = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float32)
UE4_CAMERA_TO_OPENCV = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
OPENCV_CAMERA_TO_UE4 = np.linalg.inv(UE4_CAMERA_TO_OPENCV).astype(np.float32)


@dataclass
class PoseBundle:
    world_from_sensor_carla: np.ndarray
    world_from_sensor_bevfusion: np.ndarray
    ego_from_sensor_carla: np.ndarray
    ego_from_sensor_bevfusion: np.ndarray


def ensure_numpy_matrix(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {array.shape}")
    return array


def invert_homogeneous(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(matrix).astype(np.float32)


def carla_transform_to_matrix(transform: Any) -> np.ndarray:
    return ensure_numpy_matrix(transform.get_matrix())


def carla_inverse_matrix(transform: Any) -> np.ndarray:
    return ensure_numpy_matrix(transform.get_inverse_matrix())


def carla_matrix_to_bevfusion(matrix: np.ndarray) -> np.ndarray:
    matrix = ensure_numpy_matrix(matrix)
    return (CARLA_TO_BEVFUSION @ matrix @ CARLA_TO_BEVFUSION).astype(np.float32)


def camera_world_matrix_from_carla(matrix_world_from_camera_ue4: np.ndarray) -> np.ndarray:
    matrix_world_from_camera_ue4 = ensure_numpy_matrix(matrix_world_from_camera_ue4)
    return (CARLA_TO_BEVFUSION @ matrix_world_from_camera_ue4 @ OPENCV_CAMERA_TO_UE4).astype(
        np.float32
    )


def lidar_world_matrix_from_carla(matrix_world_from_sensor_ue4: np.ndarray) -> np.ndarray:
    return carla_matrix_to_bevfusion(matrix_world_from_sensor_ue4)


def pose_bundle(
    world_from_sensor_carla: np.ndarray,
    world_from_ego_carla: np.ndarray,
    *,
    is_camera: bool,
) -> PoseBundle:
    world_from_sensor_carla = ensure_numpy_matrix(world_from_sensor_carla)
    world_from_ego_carla = ensure_numpy_matrix(world_from_ego_carla)

    ego_from_sensor_carla = invert_homogeneous(world_from_ego_carla) @ world_from_sensor_carla
    if is_camera:
        world_from_sensor_bevfusion = camera_world_matrix_from_carla(world_from_sensor_carla)
        ego_from_sensor_bevfusion = (
            carla_matrix_to_bevfusion(invert_homogeneous(world_from_ego_carla))
            @ world_from_sensor_bevfusion
        )
    else:
        world_from_sensor_bevfusion = lidar_world_matrix_from_carla(world_from_sensor_carla)
        ego_from_sensor_bevfusion = (
            carla_matrix_to_bevfusion(invert_homogeneous(world_from_ego_carla))
            @ world_from_sensor_bevfusion
        )

    return PoseBundle(
        world_from_sensor_carla=world_from_sensor_carla.astype(np.float32),
        world_from_sensor_bevfusion=world_from_sensor_bevfusion.astype(np.float32),
        ego_from_sensor_carla=ego_from_sensor_carla.astype(np.float32),
        ego_from_sensor_bevfusion=ego_from_sensor_bevfusion.astype(np.float32),
    )


def carla_vector_to_bevfusion_xyz(vector_xyz: np.ndarray) -> np.ndarray:
    vector_xyz = np.asarray(vector_xyz, dtype=np.float32)
    if vector_xyz.ndim == 1:
        return np.array([vector_xyz[0], -vector_xyz[1], vector_xyz[2]], dtype=np.float32)
    converted = vector_xyz.copy()
    converted[..., 1] *= -1.0
    return converted


def convert_lidar_points_carla_to_bevfusion(points_xyzi: np.ndarray) -> np.ndarray:
    if points_xyzi.size == 0:
        return points_xyzi.astype(np.float32).reshape(0, 4)
    converted = np.asarray(points_xyzi, dtype=np.float32).copy()
    converted[:, :3] = carla_vector_to_bevfusion_xyz(converted[:, :3])
    return converted


def append_homogeneous(points_xyz: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    return np.concatenate([points_xyz, ones], axis=1)


def transform_points(points_xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.size == 0:
        return points_xyz.reshape(0, 3)
    homogeneous = append_homogeneous(points_xyz)
    transformed = homogeneous @ ensure_numpy_matrix(matrix).T
    return transformed[:, :3].astype(np.float32)


def make_transform_json(matrix: np.ndarray) -> List[List[float]]:
    return ensure_numpy_matrix(matrix).astype(float).tolist()


def rotation_matrix_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cx, cy, cz = math.cos(roll), math.cos(pitch), math.cos(yaw)
    sx, sy, sz = math.sin(roll), math.sin(pitch), math.sin(yaw)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (rot_z @ rot_y @ rot_x).astype(np.float32)


def camera_intrinsics_from_width_height_fov(
    width: int,
    height: int,
    fov_degrees: float,
) -> np.ndarray:
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = focal
    intrinsics[1, 1] = focal
    intrinsics[0, 2] = width / 2.0
    intrinsics[1, 2] = height / 2.0
    return intrinsics


def homogeneous_camera_intrinsics(intrinsics_3x3: np.ndarray) -> np.ndarray:
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = np.asarray(intrinsics_3x3, dtype=np.float32)
    return intrinsics


def make_image_aug_matrix(
    image_width: int,
    image_height: int,
    final_dim: Sequence[int] = DEFAULT_IMAGE_SIZE,
    resize: float = 0.48,
) -> np.ndarray:
    final_h, final_w = int(final_dim[0]), int(final_dim[1])
    new_w = int(image_width * resize)
    new_h = int(image_height * resize)
    crop_h = int(new_h - final_h)
    crop_w = int(max(0, new_w - final_w) / 2)

    transform = np.eye(4, dtype=np.float32)
    transform[0, 0] = resize
    transform[1, 1] = resize
    transform[0, 3] = -float(crop_w)
    transform[1, 3] = -float(crop_h)
    return transform


def serialize_location_rotation(location: Any, rotation: Any) -> Dict[str, Dict[str, float]]:
    return {
        "location": {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        },
        "rotation": {
            "roll": float(rotation.roll),
            "pitch": float(rotation.pitch),
            "yaw": float(rotation.yaw),
        },
    }
