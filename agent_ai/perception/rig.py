from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import carla

from .constants import (
    DEFAULT_FIXED_DELTA_SECONDS,
    LIDAR_SENSOR_NAME,
    MODEL_CAMERA_ORDER,
    RADAR_SENSOR_ORDER,
    SENSOR_NAMES,
    assert_sensor_name_set,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SensorSpec:
    name: str
    blueprint: str
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    attributes: Dict[str, str] = field(default_factory=dict)

    def transform(self) -> carla.Transform:
        return carla.Transform(
            carla.Location(x=self.x, y=self.y, z=self.z),
            carla.Rotation(roll=self.roll, pitch=self.pitch, yaw=self.yaw),
        )


@dataclass(frozen=True)
class RigPreset:
    name: str
    cameras: List[SensorSpec]
    lidar: SensorSpec
    radars: List[SensorSpec]

    @property
    def all_sensors(self) -> List[SensorSpec]:
        return [*self.cameras, self.lidar, *self.radars]

    def validate(self) -> None:
        assert_sensor_name_set([sensor.name for sensor in self.all_sensors])


def default_nuscenes_like_rig(
    *,
    image_width: int = 1600,
    image_height: int = 900,
    camera_fov: float = 70.0,
    fixed_delta_seconds: float = DEFAULT_FIXED_DELTA_SECONDS,
) -> RigPreset:
    sensor_tick = f"{fixed_delta_seconds:.6f}"
    camera_attrs = {
        "image_size_x": str(image_width),
        "image_size_y": str(image_height),
        "fov": f"{camera_fov:.2f}",
        "sensor_tick": sensor_tick,
        "enable_postprocess_effects": "false",
        "motion_blur_intensity": "0.0",
        "blur_amount": "0.0",
        "chromatic_aberration_intensity": "0.0",
        "gamma": "2.2",
    }
    lidar_attrs = {
        "sensor_tick": sensor_tick,
        "channels": "32",
        "range": "120.0",
        "points_per_second": "360000",
        "rotation_frequency": f"{1.0 / fixed_delta_seconds:.6f}",
        "upper_fov": "10.0",
        "lower_fov": "-30.0",
        "dropoff_general_rate": "0.0",
        "dropoff_intensity_limit": "1.0",
        "dropoff_zero_intensity": "0.0",
    }
    radar_attrs = {
        "sensor_tick": sensor_tick,
        "horizontal_fov": "35.0",
        "vertical_fov": "20.0",
        "range": "120.0",
        "points_per_second": "1500",
    }

    cameras = [
        SensorSpec("CAM_FRONT", "sensor.camera.rgb", 1.55, 0.00, 1.70, yaw=0.0, attributes=dict(camera_attrs)),
        SensorSpec("CAM_FRONT_RIGHT", "sensor.camera.rgb", 1.30, 0.55, 1.70, yaw=55.0, attributes=dict(camera_attrs)),
        SensorSpec("CAM_FRONT_LEFT", "sensor.camera.rgb", 1.30, -0.55, 1.70, yaw=-55.0, attributes=dict(camera_attrs)),
        SensorSpec("CAM_BACK", "sensor.camera.rgb", -1.55, 0.00, 1.70, yaw=180.0, attributes=dict(camera_attrs)),
        SensorSpec("CAM_BACK_LEFT", "sensor.camera.rgb", -1.30, -0.55, 1.70, yaw=-125.0, attributes=dict(camera_attrs)),
        SensorSpec("CAM_BACK_RIGHT", "sensor.camera.rgb", -1.30, 0.55, 1.70, yaw=125.0, attributes=dict(camera_attrs)),
    ]
    lidar = SensorSpec(
        name=LIDAR_SENSOR_NAME,
        blueprint="sensor.lidar.ray_cast",
        x=0.0,
        y=0.0,
        z=2.20,
        attributes=lidar_attrs,
    )
    radars = [
        SensorSpec("RADAR_FRONT", "sensor.other.radar", 2.00, 0.00, 0.90, yaw=0.0, attributes=dict(radar_attrs)),
        SensorSpec("RADAR_FRONT_LEFT", "sensor.other.radar", 1.65, -0.75, 0.90, yaw=-45.0, attributes=dict(radar_attrs)),
        SensorSpec("RADAR_FRONT_RIGHT", "sensor.other.radar", 1.65, 0.75, 0.90, yaw=45.0, attributes=dict(radar_attrs)),
        SensorSpec("RADAR_BACK_LEFT", "sensor.other.radar", -1.65, -0.75, 0.90, yaw=-135.0, attributes=dict(radar_attrs)),
        SensorSpec("RADAR_BACK_RIGHT", "sensor.other.radar", -1.65, 0.75, 0.90, yaw=135.0, attributes=dict(radar_attrs)),
    ]
    preset = RigPreset(name="nuscenes_like_default", cameras=cameras, lidar=lidar, radars=radars)
    preset.validate()
    return preset


def low_memory_shadow_rig(
    *,
    image_width: int = 640,
    image_height: int = 360,
    camera_fov: float = 70.0,
    fixed_delta_seconds: float = DEFAULT_FIXED_DELTA_SECONDS,
) -> RigPreset:
    sensor_tick = f"{fixed_delta_seconds:.6f}"
    camera_attrs = {
        "image_size_x": str(image_width),
        "image_size_y": str(image_height),
        "fov": f"{camera_fov:.2f}",
        "sensor_tick": sensor_tick,
        "enable_postprocess_effects": "false",
        "motion_blur_intensity": "0.0",
        "blur_amount": "0.0",
        "chromatic_aberration_intensity": "0.0",
        "gamma": "2.2",
    }
    # Keep the full sensor topology so Stage 1/4 dump paths and downstream contracts
    # remain intact, but reduce per-sensor memory/throughput for six-case bring-up.
    lidar_attrs = {
        "sensor_tick": sensor_tick,
        "channels": "16",
        "range": "100.0",
        "points_per_second": "90000",
        "rotation_frequency": f"{1.0 / fixed_delta_seconds:.6f}",
        "upper_fov": "10.0",
        "lower_fov": "-30.0",
        "dropoff_general_rate": "0.0",
        "dropoff_intensity_limit": "1.0",
        "dropoff_zero_intensity": "0.0",
    }
    radar_attrs = {
        "sensor_tick": sensor_tick,
        "horizontal_fov": "30.0",
        "vertical_fov": "15.0",
        "range": "100.0",
        "points_per_second": "500",
    }

    cameras = [
        SensorSpec("CAM_FRONT", "sensor.camera.rgb", 1.55, 0.00, 1.70, yaw=0.0, attributes=dict(camera_attrs)),
        SensorSpec("CAM_FRONT_RIGHT", "sensor.camera.rgb", 1.30, 0.55, 1.70, yaw=55.0, attributes=dict(camera_attrs)),
        SensorSpec("CAM_FRONT_LEFT", "sensor.camera.rgb", 1.30, -0.55, 1.70, yaw=-55.0, attributes=dict(camera_attrs)),
        SensorSpec("CAM_BACK", "sensor.camera.rgb", -1.55, 0.00, 1.70, yaw=180.0, attributes=dict(camera_attrs)),
        SensorSpec("CAM_BACK_LEFT", "sensor.camera.rgb", -1.30, -0.55, 1.70, yaw=-125.0, attributes=dict(camera_attrs)),
        SensorSpec("CAM_BACK_RIGHT", "sensor.camera.rgb", -1.30, 0.55, 1.70, yaw=125.0, attributes=dict(camera_attrs)),
    ]
    lidar = SensorSpec(
        name=LIDAR_SENSOR_NAME,
        blueprint="sensor.lidar.ray_cast",
        x=0.0,
        y=0.0,
        z=2.20,
        attributes=lidar_attrs,
    )
    radars = [
        SensorSpec("RADAR_FRONT", "sensor.other.radar", 2.00, 0.00, 0.90, yaw=0.0, attributes=dict(radar_attrs)),
        SensorSpec("RADAR_FRONT_LEFT", "sensor.other.radar", 1.65, -0.75, 0.90, yaw=-45.0, attributes=dict(radar_attrs)),
        SensorSpec("RADAR_FRONT_RIGHT", "sensor.other.radar", 1.65, 0.75, 0.90, yaw=45.0, attributes=dict(radar_attrs)),
        SensorSpec("RADAR_BACK_LEFT", "sensor.other.radar", -1.65, -0.75, 0.90, yaw=-135.0, attributes=dict(radar_attrs)),
        SensorSpec("RADAR_BACK_RIGHT", "sensor.other.radar", -1.65, 0.75, 0.90, yaw=135.0, attributes=dict(radar_attrs)),
    ]
    preset = RigPreset(name="low_memory_shadow", cameras=cameras, lidar=lidar, radars=radars)
    preset.validate()
    return preset


def build_rig_preset(
    profile: str,
    *,
    image_width: int,
    image_height: int,
    camera_fov: float,
    fixed_delta_seconds: float,
) -> RigPreset:
    normalized = str(profile or "default").strip().lower()
    if normalized == "low_memory_shadow":
        return low_memory_shadow_rig(
            image_width=image_width,
            image_height=image_height,
            camera_fov=camera_fov,
            fixed_delta_seconds=fixed_delta_seconds,
        )
    return default_nuscenes_like_rig(
        image_width=image_width,
        image_height=image_height,
        camera_fov=camera_fov,
        fixed_delta_seconds=fixed_delta_seconds,
    )


def build_sensor_actor(
    world: carla.World,
    parent: carla.Actor,
    spec: SensorSpec,
) -> carla.Actor:
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find(spec.blueprint)
    for key, value in spec.attributes.items():
        if not blueprint.has_attribute(key):
            raise ValueError(
                f"Blueprint '{spec.blueprint}' does not expose attribute '{key}' for sensor '{spec.name}'"
            )
        blueprint.set_attribute(key, value)
    actor = world.spawn_actor(blueprint, spec.transform(), attach_to=parent)
    LOGGER.info("Spawned sensor %s (%s)", spec.name, spec.blueprint)
    return actor


def spawn_rig(
    world: carla.World,
    ego_vehicle: carla.Actor,
    preset: RigPreset,
) -> Dict[str, carla.Actor]:
    preset.validate()
    actors: Dict[str, carla.Actor] = {}
    for spec in preset.all_sensors:
        actors[spec.name] = build_sensor_actor(world, ego_vehicle, spec)
    return actors


def destroy_actors(actors: Iterable[carla.Actor]) -> None:
    for actor in actors:
        try:
            actor.destroy()
        except RuntimeError:
            LOGGER.warning("Failed to destroy actor id=%s", getattr(actor, "id", "unknown"))


def blueprint_attributes_dict(actor: carla.Actor) -> Dict[str, str]:
    return {key: value for key, value in actor.attributes.items()}
