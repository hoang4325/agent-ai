from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carla_bevfusion_stage1.adapter import load_sample_meta, radar_native_to_cartesian as radar_native_to_cartesian_bev  # noqa: E402
from carla_bevfusion_stage1.constants import DEFAULT_POINT_CLOUD_RANGE, LIDAR_SENSOR_NAME, RADAR_SENSOR_ORDER  # noqa: E402
from carla_bevfusion_stage1.coordinate_utils import convert_lidar_points_carla_to_bevfusion, transform_points  # noqa: E402
from carla_bevfusion_stage1.visualization import save_geometry_sanity_bundle, save_geometry_topdown as save_geometry_topdown_plot  # noqa: E402

LOGGER = logging.getLogger("geometry_sanity_check")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline CARLA radar/LiDAR geometry sanity checks.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--color-by", choices=("sensor", "velocity"), default="sensor")
    parser.add_argument(
        "--point-cloud-range",
        type=float,
        nargs=6,
        default=DEFAULT_POINT_CLOUD_RANGE,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_dump_sample(sample_dir: Path) -> Dict[str, Any]:
    meta = load_sample_meta(sample_dir)
    lidar_raw = np.load(sample_dir / "lidar" / f"{LIDAR_SENSOR_NAME}.npy")
    radar_raw = {
        sensor_name: np.load(sample_dir / "radar" / f"{sensor_name}.npy")
        for sensor_name in RADAR_SENSOR_ORDER
    }
    return {
        "meta": meta,
        "lidar_raw": lidar_raw,
        "radar_raw": radar_raw,
    }


def radar_native_to_cartesian(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz_sensor_bev, radial_velocity, los_unit_sensor_bev = radar_native_to_cartesian_bev(raw)
    return xyz_sensor_bev, radial_velocity, los_unit_sensor_bev[:, :2]


def transform_sensor_points_to_ego_bev(meta: Dict[str, Any], sensor_name: str, xyz_sensor_bev: np.ndarray) -> np.ndarray:
    matrix_ego_from_sensor = np.asarray(
        meta["sensors"][sensor_name]["matrix_ego_from_sensor_bevfusion"], dtype=np.float32
    )
    return transform_points(xyz_sensor_bev, matrix_ego_from_sensor)


def save_geometry_topdown(
    lidar_bev: np.ndarray,
    radar_by_sensor: Mapping[str, np.ndarray],
    output_png: Path,
    color_by: str,
    velocity_by_sensor: Mapping[str, np.ndarray] | None = None,
) -> None:
    save_geometry_topdown_plot(
        output_path=output_png,
        lidar_points=lidar_bev,
        radar_by_sensor=radar_by_sensor,
        velocity_by_sensor=velocity_by_sensor,
        color_by=color_by,
    )


def _range_mask_bev(points_xyz: np.ndarray, point_cloud_range: Tuple[float, ...]) -> np.ndarray:
    x_min, y_min, _z_min, x_max, y_max, _z_max = point_cloud_range
    return (
        (points_xyz[:, 0] > x_min)
        & (points_xyz[:, 1] > y_min)
        & (points_xyz[:, 0] < x_max)
        & (points_xyz[:, 1] < y_max)
    )


def _range_mask_xyz(points_xyz: np.ndarray, point_cloud_range: Tuple[float, ...]) -> np.ndarray:
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    return (
        (points_xyz[:, 0] > x_min)
        & (points_xyz[:, 1] > y_min)
        & (points_xyz[:, 2] > z_min)
        & (points_xyz[:, 0] < x_max)
        & (points_xyz[:, 1] < y_max)
        & (points_xyz[:, 2] < z_max)
    )


def _sensor_direction_expectation(sensor_name: str, mean_xyz: np.ndarray) -> str | None:
    warnings = []
    if "FRONT" in sensor_name and mean_xyz[0] <= 0:
        warnings.append(f"{sensor_name} expected x>0 but mean x={mean_xyz[0]:.3f}")
    if "BACK" in sensor_name and mean_xyz[0] >= 0:
        warnings.append(f"{sensor_name} expected x<0 but mean x={mean_xyz[0]:.3f}")
    if sensor_name.endswith("LEFT") and mean_xyz[1] <= 0:
        warnings.append(f"{sensor_name} expected y>0 but mean y={mean_xyz[1]:.3f}")
    if sensor_name.endswith("RIGHT") and mean_xyz[1] >= 0:
        warnings.append(f"{sensor_name} expected y<0 but mean y={mean_xyz[1]:.3f}")
    return "; ".join(warnings) if warnings else None


def build_geometry_report(sample: Dict[str, Any], point_cloud_range: Tuple[float, ...]) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    meta = sample["meta"]
    lidar_bev = convert_lidar_points_carla_to_bevfusion(sample["lidar_raw"])
    radar_by_sensor: Dict[str, np.ndarray] = {}
    velocity_by_sensor: Dict[str, np.ndarray] = {}
    warnings = []

    radar_report: Dict[str, Any] = {}
    raw_total = 0
    bev_total = 0
    xyz_total = 0

    for sensor_name in RADAR_SENSOR_ORDER:
        xyz_sensor_bev, radial_velocity, _ = radar_native_to_cartesian(sample["radar_raw"][sensor_name])
        xyz_ego_bev = transform_sensor_points_to_ego_bev(meta, sensor_name, xyz_sensor_bev)
        radar_by_sensor[sensor_name] = xyz_ego_bev
        velocity_by_sensor[sensor_name] = radial_velocity

        ranges = np.linalg.norm(xyz_sensor_bev, axis=1) if len(xyz_sensor_bev) else np.zeros((0,), dtype=np.float32)
        bev_mask = _range_mask_bev(xyz_ego_bev, point_cloud_range) if len(xyz_ego_bev) else np.zeros((0,), dtype=bool)
        xyz_mask = _range_mask_xyz(xyz_ego_bev, point_cloud_range) if len(xyz_ego_bev) else np.zeros((0,), dtype=bool)
        mean_xyz = xyz_ego_bev.mean(axis=0) if len(xyz_ego_bev) else np.zeros((3,), dtype=np.float32)

        raw_total += int(xyz_ego_bev.shape[0])
        bev_total += int(bev_mask.sum())
        xyz_total += int(xyz_mask.sum())

        direction_warning = _sensor_direction_expectation(sensor_name, mean_xyz)
        if direction_warning:
            warnings.append(direction_warning)

        radar_report[sensor_name] = {
            "count": int(xyz_ego_bev.shape[0]),
            "range_min": float(ranges.min()) if len(ranges) else 0.0,
            "range_max": float(ranges.max()) if len(ranges) else 0.0,
            "velocity_min": float(radial_velocity.min()) if len(radial_velocity) else 0.0,
            "velocity_max": float(radial_velocity.max()) if len(radial_velocity) else 0.0,
            "ego_bev_min_xyz": xyz_ego_bev.min(axis=0).astype(float).tolist() if len(xyz_ego_bev) else [0.0, 0.0, 0.0],
            "ego_bev_max_xyz": xyz_ego_bev.max(axis=0).astype(float).tolist() if len(xyz_ego_bev) else [0.0, 0.0, 0.0],
            "ego_bev_mean_xyz": mean_xyz.astype(float).tolist(),
            "in_bev_xy_count": int(bev_mask.sum()),
            "in_full_xyz_count": int(xyz_mask.sum()),
        }

    if len(lidar_bev):
        lidar_report = {
            "count": int(lidar_bev.shape[0]),
            "ego_bev_min_xyz": lidar_bev[:, :3].min(axis=0).astype(float).tolist(),
            "ego_bev_max_xyz": lidar_bev[:, :3].max(axis=0).astype(float).tolist(),
        }
    else:
        lidar_report = {
            "count": 0,
            "ego_bev_min_xyz": [0.0, 0.0, 0.0],
            "ego_bev_max_xyz": [0.0, 0.0, 0.0],
        }

    z_out_of_range = max(0, bev_total - xyz_total)
    if z_out_of_range > 0:
        warnings.append(
            f"Radar z spread exceeds voxel range: {z_out_of_range} points are inside XY BEV range but outside full XYZ voxel range."
        )

    report = {
        "sample_name": meta["sample_name"],
        "frame_id": meta["frame_id"],
        "timestamp": meta["timestamp"],
        "town": meta.get("town"),
        "point_cloud_range": list(point_cloud_range),
        "geometry_pass": len([item for item in warnings if "expected" in item]) == 0,
        "warnings": warnings,
        "lidar": lidar_report,
        "radar": radar_report,
        "radar_raw_total": raw_total,
        "radar_in_bev_xy_total": bev_total,
        "radar_in_full_xyz_total": xyz_total,
        "radar_z_out_of_range_total": z_out_of_range,
    }
    return report, lidar_bev[:, :3], radar_by_sensor, velocity_by_sensor


def main() -> None:
    args = parse_args()
    configure_logging()

    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    point_cloud_range = tuple(float(value) for value in args.point_cloud_range)

    sample = load_dump_sample(sample_dir)
    report, lidar_bev, radar_by_sensor, velocity_by_sensor = build_geometry_report(sample, point_cloud_range)
    save_geometry_sanity_bundle(
        output_root=output_dir,
        lidar_points=lidar_bev,
        radar_by_sensor=radar_by_sensor,
        velocity_by_sensor=velocity_by_sensor,
        report=report,
        color_by=args.color_by,
    )

    LOGGER.info(
        "Geometry sanity complete for %s frame=%s radar_raw_total=%d radar_in_bev_xy_total=%d radar_in_full_xyz_total=%d",
        report["sample_name"],
        report["frame_id"],
        report["radar_raw_total"],
        report["radar_in_bev_xy_total"],
        report["radar_in_full_xyz_total"],
    )


if __name__ == "__main__":
    main()
