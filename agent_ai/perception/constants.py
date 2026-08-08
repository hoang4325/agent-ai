from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

MODEL_CAMERA_ORDER: Tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

RADAR_SENSOR_ORDER: Tuple[str, ...] = (
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
)

LIDAR_SENSOR_NAME = "LIDAR_TOP"

SENSOR_NAMES: Tuple[str, ...] = MODEL_CAMERA_ORDER + (LIDAR_SENSOR_NAME,) + RADAR_SENSOR_ORDER

DEFAULT_OBJECT_CLASSES: Tuple[str, ...] = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)

DEFAULT_FIXED_DELTA_SECONDS = 0.05
DEFAULT_IMAGE_SIZE = (256, 704)
DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)
DEFAULT_POINT_CLOUD_RANGE = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)

RADAR_USE_DIMS: Tuple[int, ...] = (
    0,
    1,
    2,
    5,
    8,
    9,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
)

RADAR_BASE_COLUMNS: Tuple[str, ...] = (
    "x",
    "y",
    "z",
    "dyn_prop",
    "id",
    "rcs",
    "vx",
    "vy",
    "vx_comp",
    "vy_comp",
    "is_quality_valid",
    "ambig_state",
    "x_rms",
    "y_rms",
    "invalid_state",
    "pdh0",
    "vx_rms",
    "vy_rms",
    "time_diff",
)

RADAR_ENCODED_COLUMNS: Tuple[str, ...] = (
    *(f"dynprop_oh_{idx}" for idx in range(8)),
    *(f"ambig_state_oh_{idx}" for idx in range(5)),
    *(f"invalid_state_oh_{idx}" for idx in range(18)),
    *(f"pdh0_ord_gt_{idx}" for idx in range(7)),
    "nusc_filter_flag",
)

RADAR_ALL_COLUMNS: Tuple[str, ...] = RADAR_BASE_COLUMNS + RADAR_ENCODED_COLUMNS
RADAR_TRACE_SELECTED_COLUMNS: Tuple[str, ...] = tuple(RADAR_ALL_COLUMNS[idx] for idx in RADAR_USE_DIMS)

MINIMAL_RADAR_COLUMNS: Tuple[str, ...] = (
    "x",
    "y",
    "z",
    "vx_comp",
    "vy_comp",
    "time_diff",
)

# RadarFeatureNet adds f_center_x/f_center_y internally before the first linear
# layer, so checkpoint slicing keeps those trailing columns as well.
MINIMAL_RADAR_CHECKPOINT_COLUMN_INDICES: Tuple[int, ...] = (
    0,
    1,
    2,
    4,
    5,
    6,
    45,
    46,
)

MINIMAL_RADAR_CHECKPOINT_COLUMN_NAMES: Tuple[str, ...] = (
    "x",
    "y",
    "z",
    "vx_comp",
    "vy_comp",
    "time_diff",
    "f_center_x",
    "f_center_y",
)

CARLA_RADAR_RAW_LAYOUT: Tuple[str, ...] = (
    "velocity",
    "altitude",
    "azimuth",
    "depth",
)

CARLA_LIDAR_RAW_LAYOUT: Tuple[str, ...] = (
    "x",
    "y",
    "z",
    "intensity",
)

CARLA_IMAGE_RAW_LAYOUT = "bgra8"


def missing_sensor_names(names: Iterable[str]) -> List[str]:
    expected = set(SENSOR_NAMES)
    actual = set(names)
    return sorted(expected - actual)


def unexpected_sensor_names(names: Iterable[str]) -> List[str]:
    expected = set(SENSOR_NAMES)
    actual = set(names)
    return sorted(actual - expected)


def assert_sensor_name_set(names: Sequence[str]) -> None:
    missing = missing_sensor_names(names)
    unexpected = unexpected_sensor_names(names)
    if missing or unexpected:
        raise ValueError(
            f"sensor name mismatch: missing={missing or '[]'} unexpected={unexpected or '[]'}"
        )
