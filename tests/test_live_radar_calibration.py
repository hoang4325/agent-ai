from __future__ import annotations

import sys
import types
import unittest

import numpy as np

# The unit-test host does not need the CARLA wheel. The modules under test only
# require it when actors are actually spawned.
sys.modules.setdefault("carla", types.ModuleType("carla"))

from carla_bevfusion_stage1.carla_sensor_sync import LiveCalibration, _build_calibration
from carla_bevfusion_stage1.constants import (
    LIDAR_SENSOR_NAME,
    MODEL_CAMERA_ORDER,
    RADAR_SENSOR_ORDER,
)


class _Transform:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def get_matrix(self):
        return self._matrix.tolist()


class _Actor:
    def __init__(self, matrix: np.ndarray) -> None:
        self._transform = _Transform(matrix)

    def get_transform(self) -> _Transform:
        return self._transform


class LiveRadarCalibrationTests(unittest.TestCase):
    def test_live_calibration_contains_every_radar_extrinsic(self) -> None:
        identity = np.eye(4, dtype=np.float32)
        actors = {
            name: _Actor(identity.copy())
            for name in (*MODEL_CAMERA_ORDER, LIDAR_SENSOR_NAME, *RADAR_SENSOR_ORDER)
        }

        calibration = _build_calibration(
            ego_actor=_Actor(identity.copy()),
            sensor_actors=actors,
            preset=None,
            image_width=1600,
            image_height=900,
            camera_fov=70.0,
        )

        self.assertEqual(set(calibration.world_from_radars_bev), set(RADAR_SENSOR_ORDER))
        self.assertEqual(set(calibration.ego_from_radars_bev), set(RADAR_SENSOR_ORDER))

    def test_live_radar_points_are_transformed_instead_of_discarded(self) -> None:
        try:
            from carla_bevfusion_stage1.bevfusion_live_adapter import _build_live_radar
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest("torch is not installed on the host test environment")
            raise

        identity = np.eye(4, dtype=np.float32)
        radar_from_world = identity.copy()
        radar_from_world[0, 3] = 2.0
        calibration = LiveCalibration(
            world_from_lidar_bev=identity,
            ego_from_lidar_bev=identity,
            world_from_cameras_bev={},
            ego_from_cameras_bev={},
            camera_intrinsics={},
            world_from_radars_bev={"RADAR_FRONT": radar_from_world},
        )
        # CARLA columns: radial velocity, altitude, azimuth, depth.
        radar_raw = {
            "RADAR_FRONT": np.array([[2.0, 0.0, 0.0, 10.0]], dtype=np.float32)
        }

        points = _build_live_radar(
            radar_raw,
            calibration,
            (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        )

        self.assertEqual(points.shape, (1, 6))
        np.testing.assert_allclose(points[0], [12.0, 0.0, 0.0, 2.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
