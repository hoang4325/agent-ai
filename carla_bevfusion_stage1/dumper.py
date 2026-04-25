from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
from PIL import Image

import carla

from .collector import FrameCapture
from .constants import (
    CARLA_IMAGE_RAW_LAYOUT,
    CARLA_LIDAR_RAW_LAYOUT,
    CARLA_RADAR_RAW_LAYOUT,
    LIDAR_SENSOR_NAME,
    MODEL_CAMERA_ORDER,
    RADAR_SENSOR_ORDER,
)
from .coordinate_utils import (
    camera_intrinsics_from_width_height_fov,
    carla_transform_to_matrix,
    carla_matrix_to_bevfusion,
    make_transform_json,
    pose_bundle,
    serialize_location_rotation,
)
from .rig import RigPreset, blueprint_attributes_dict

LOGGER = logging.getLogger(__name__)


def decode_carla_rgb(image: carla.Image) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3][:, :, ::-1].copy()


def decode_carla_lidar(lidar: carla.LidarMeasurement) -> np.ndarray:
    points = np.frombuffer(lidar.raw_data, dtype=np.float32)
    return points.reshape((-1, 4)).copy()


def decode_carla_radar(radar: carla.RadarMeasurement) -> np.ndarray:
    detections = np.frombuffer(radar.raw_data, dtype=np.float32)
    return detections.reshape((len(radar), 4)).copy()


def weather_to_dict(weather: carla.WeatherParameters) -> Dict[str, float]:
    payload: Dict[str, float] = {}
    for key in dir(weather):
        if key.startswith("_"):
            continue
        value = getattr(weather, key)
        if isinstance(value, (float, int)):
            payload[key] = float(value)
    return payload


@dataclass
class DumpResult:
    sample_dir: Path
    meta_path: Path


class SampleDumper:
    def __init__(
        self,
        output_root: str | Path,
        rig_preset: RigPreset,
        sensor_actors: Mapping[str, carla.Actor],
    ) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.rig_preset = rig_preset
        self.sensor_actors = dict(sensor_actors)

    def dump_capture(
        self,
        capture: FrameCapture,
        *,
        ego_vehicle: carla.Actor,
        world: carla.World,
        sample_index: int,
        previous_sample_name: str | None = None,
    ) -> DumpResult:
        sample_name = f"sample_{sample_index:06d}"
        sample_dir = self.output_root / sample_name
        images_dir = sample_dir / "images"
        lidar_dir = sample_dir / "lidar"
        radar_dir = sample_dir / "radar"
        for directory in (images_dir, lidar_dir, radar_dir):
            directory.mkdir(parents=True, exist_ok=True)

        ego_transform = ego_vehicle.get_transform()
        world_from_ego_carla = carla_transform_to_matrix(ego_transform)
        world_from_ego_bevfusion = carla_matrix_to_bevfusion(world_from_ego_carla)

        sensor_meta: Dict[str, Any] = {}
        frame_log: Dict[str, Any] = {}

        for camera_name in MODEL_CAMERA_ORDER:
            packet = capture.packets[camera_name]
            image = packet.measurement
            rgb = decode_carla_rgb(image)
            output_path = images_dir / f"{camera_name}.png"
            Image.fromarray(rgb).save(output_path)

            sensor_transform = image.transform
            sensor_pose = pose_bundle(
                world_from_sensor_carla=carla_transform_to_matrix(sensor_transform),
                world_from_ego_carla=world_from_ego_carla,
                is_camera=True,
            )
            sensor_actor = self.sensor_actors[camera_name]
            intrinsics = camera_intrinsics_from_width_height_fov(
                width=int(sensor_actor.attributes["image_size_x"]),
                height=int(sensor_actor.attributes["image_size_y"]),
                fov_degrees=float(sensor_actor.attributes["fov"]),
            )

            sensor_meta[camera_name] = {
                "type_id": sensor_actor.type_id,
                "blueprint_attributes": blueprint_attributes_dict(sensor_actor),
                "transform_carla": serialize_location_rotation(
                    sensor_transform.location, sensor_transform.rotation
                ),
                "matrix_world_from_sensor_carla": make_transform_json(sensor_pose.world_from_sensor_carla),
                "matrix_world_from_sensor_bevfusion": make_transform_json(
                    sensor_pose.world_from_sensor_bevfusion
                ),
                "matrix_ego_from_sensor_carla": make_transform_json(sensor_pose.ego_from_sensor_carla),
                "matrix_ego_from_sensor_bevfusion": make_transform_json(
                    sensor_pose.ego_from_sensor_bevfusion
                ),
                "camera_intrinsics": intrinsics.astype(float).tolist(),
                "data_path": str(output_path),
                "raw_layout": CARLA_IMAGE_RAW_LAYOUT,
            }
            frame_log[camera_name] = {
                "frame": packet.frame,
                "timestamp": packet.timestamp,
                "shape": [int(image.height), int(image.width), 3],
                "mean_rgb": rgb.mean(axis=(0, 1)).astype(float).tolist(),
            }

        lidar_packet = capture.packets[LIDAR_SENSOR_NAME]
        lidar_measurement = lidar_packet.measurement
        lidar_array = decode_carla_lidar(lidar_measurement)
        lidar_output = lidar_dir / f"{LIDAR_SENSOR_NAME}.npy"
        np.save(lidar_output, lidar_array)
        lidar_pose = pose_bundle(
            world_from_sensor_carla=carla_transform_to_matrix(lidar_measurement.transform),
            world_from_ego_carla=world_from_ego_carla,
            is_camera=False,
        )
        sensor_meta[LIDAR_SENSOR_NAME] = {
            "type_id": self.sensor_actors[LIDAR_SENSOR_NAME].type_id,
            "blueprint_attributes": blueprint_attributes_dict(self.sensor_actors[LIDAR_SENSOR_NAME]),
            "transform_carla": serialize_location_rotation(
                lidar_measurement.transform.location, lidar_measurement.transform.rotation
            ),
            "matrix_world_from_sensor_carla": make_transform_json(lidar_pose.world_from_sensor_carla),
            "matrix_world_from_sensor_bevfusion": make_transform_json(
                lidar_pose.world_from_sensor_bevfusion
            ),
            "matrix_ego_from_sensor_carla": make_transform_json(lidar_pose.ego_from_sensor_carla),
            "matrix_ego_from_sensor_bevfusion": make_transform_json(
                lidar_pose.ego_from_sensor_bevfusion
            ),
            "data_path": str(lidar_output),
            "raw_layout": list(CARLA_LIDAR_RAW_LAYOUT),
        }
        frame_log[LIDAR_SENSOR_NAME] = {
            "frame": lidar_packet.frame,
            "timestamp": lidar_packet.timestamp,
            "point_count": int(lidar_array.shape[0]),
            "min_xyz": lidar_array[:, :3].min(axis=0).astype(float).tolist() if len(lidar_array) else [0, 0, 0],
            "max_xyz": lidar_array[:, :3].max(axis=0).astype(float).tolist() if len(lidar_array) else [0, 0, 0],
        }

        for radar_name in RADAR_SENSOR_ORDER:
            packet = capture.packets[radar_name]
            radar_measurement = packet.measurement
            radar_array = decode_carla_radar(radar_measurement)
            radar_output = radar_dir / f"{radar_name}.npy"
            np.save(radar_output, radar_array)

            radar_pose = pose_bundle(
                world_from_sensor_carla=carla_transform_to_matrix(radar_measurement.transform),
                world_from_ego_carla=world_from_ego_carla,
                is_camera=False,
            )
            sensor_meta[radar_name] = {
                "type_id": self.sensor_actors[radar_name].type_id,
                "blueprint_attributes": blueprint_attributes_dict(self.sensor_actors[radar_name]),
                "transform_carla": serialize_location_rotation(
                    radar_measurement.transform.location, radar_measurement.transform.rotation
                ),
                "matrix_world_from_sensor_carla": make_transform_json(radar_pose.world_from_sensor_carla),
                "matrix_world_from_sensor_bevfusion": make_transform_json(
                    radar_pose.world_from_sensor_bevfusion
                ),
                "matrix_ego_from_sensor_carla": make_transform_json(radar_pose.ego_from_sensor_carla),
                "matrix_ego_from_sensor_bevfusion": make_transform_json(
                    radar_pose.ego_from_sensor_bevfusion
                ),
                "data_path": str(radar_output),
                "raw_layout": list(CARLA_RADAR_RAW_LAYOUT),
            }
            frame_log[radar_name] = {
                "frame": packet.frame,
                "timestamp": packet.timestamp,
                "point_count": int(radar_array.shape[0]),
            }

        meta = {
            "schema_version": "stage1.v1",
            "sample_name": sample_name,
            "sequence_index": sample_index,
            "previous_sample_name": previous_sample_name,
            "frame_id": capture.frame,
            "timestamp": float(capture.world_snapshot.timestamp.elapsed_seconds),
            "platform_timestamp": float(getattr(capture.world_snapshot.timestamp, "platform_timestamp", 0.0)),
            "delta_seconds": float(getattr(capture.world_snapshot.timestamp, "delta_seconds", 0.0)),
            "town": world.get_map().name,
            "weather": weather_to_dict(world.get_weather()),
            "sensor_names": list(self.sensor_actors.keys()),
            "camera_order_for_model": list(MODEL_CAMERA_ORDER),
            "radar_order_for_model": list(RADAR_SENSOR_ORDER),
            "coordinate_notes": {
                "carla_world": "left-handed x-forward y-right z-up",
                "bevfusion_lidar": "x-forward y-left z-up",
                "camera_frame_for_model": "OpenCV style x-right y-down z-forward",
            },
            "ego": {
                "actor_id": int(ego_vehicle.id),
                "transform_carla": serialize_location_rotation(
                    ego_transform.location, ego_transform.rotation
                ),
                "velocity_carla": [
                    float(ego_vehicle.get_velocity().x),
                    float(ego_vehicle.get_velocity().y),
                    float(ego_vehicle.get_velocity().z),
                ],
                "angular_velocity_carla": [
                    float(ego_vehicle.get_angular_velocity().x),
                    float(ego_vehicle.get_angular_velocity().y),
                    float(ego_vehicle.get_angular_velocity().z),
                ],
                "matrix_world_from_ego_carla": make_transform_json(world_from_ego_carla),
                "matrix_world_from_ego_bevfusion": make_transform_json(world_from_ego_bevfusion),
            },
            "sensors": sensor_meta,
            "frame_log": frame_log,
        }

        meta_path = sample_dir / "meta.json"
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        complete_marker_path = sample_dir / "sample_complete.json"
        marker = {
            "sample_name": sample_name,
            "frame_id": int(capture.frame),
            "meta_path": str(meta_path),
            "ready": True,
        }
        with complete_marker_path.open("w", encoding="utf-8") as handle:
            json.dump(marker, handle, indent=2)
        LOGGER.info(
            "Dumped sample %s frame=%d cameras=%d lidar_pts=%d radar_pts=%d",
            sample_name,
            capture.frame,
            len(MODEL_CAMERA_ORDER),
            int(lidar_array.shape[0]),
            sum(int(frame_log[name]["point_count"]) for name in RADAR_SENSOR_ORDER),
        )
        return DumpResult(sample_dir=sample_dir, meta_path=meta_path)
