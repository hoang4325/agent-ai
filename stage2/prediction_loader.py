from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from carla_bevfusion_stage1.constants import DEFAULT_OBJECT_CLASSES

from .object_schema import EgoState, NormalizedDetection, NormalizedFramePrediction

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PREFIX = "/workspace/Agent-AI"

CLASS_GROUPS: Dict[str, str] = {
    "car": "vehicle",
    "truck": "vehicle",
    "construction_vehicle": "vehicle",
    "bus": "vehicle",
    "trailer": "vehicle",
    "barrier": "static",
    "motorcycle": "vru",
    "bicycle": "vru",
    "pedestrian": "vru",
    "traffic_cone": "static",
}


@dataclass
class Stage1FrameArtifact:
    sample_name: str
    sequence_index: int
    frame_id: int
    timestamp: float
    source_sample_dir: Path
    source_prediction_dir: Path
    prediction_variant: str
    stage1_status: str


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_artifact_path(raw_path: str | None, *, session_dir: Path | None = None) -> Path:
    if not raw_path:
        raise ValueError("Missing artifact path in stage1 session logs.")

    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    if session_dir is not None and not candidate.is_absolute():
        relative_candidate = (session_dir / candidate).resolve()
        if relative_candidate.exists():
            return relative_candidate

    normalized = raw_path.replace("\\", "/")
    if normalized.startswith(WORKSPACE_PREFIX):
        relative = normalized[len(WORKSPACE_PREFIX) :].lstrip("/")
        mapped = PROJECT_ROOT / Path(relative)
        if mapped.exists():
            return mapped

    if normalized.startswith("/workspace/"):
        workspace_parts = normalized.split("/", 3)
        if len(workspace_parts) >= 4:
            mapped = PROJECT_ROOT.parent / workspace_parts[3]
            if mapped.exists():
                return mapped

    raise FileNotFoundError(f"Could not resolve artifact path: {raw_path}")


def _prediction_variant_from_status(status: str, requested_variant: str) -> str:
    if status == "inferred_compare":
        if requested_variant == "auto":
            return "bridge_minimal"
        return requested_variant
    if status == "inferred":
        return "single"
    raise ValueError(f"Unsupported stage1 status for prediction loading: {status}")


def list_stage1_frame_artifacts(
    stage1_session_dir: str | Path,
    *,
    prediction_variant: str = "auto",
) -> List[Stage1FrameArtifact]:
    session_dir = Path(stage1_session_dir)
    summary_path = session_dir / "live_summary.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing live_summary.jsonl in {session_dir}")

    artifacts: List[Stage1FrameArtifact] = []
    with summary_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            status = payload.get("status")
            if status not in {"inferred", "inferred_compare"}:
                continue

            sample_dir = _resolve_artifact_path(payload.get("source_sample_dir"), session_dir=session_dir)
            chosen_variant = _prediction_variant_from_status(status, prediction_variant)
            if status == "inferred_compare":
                compare_root = _resolve_artifact_path(payload.get("comparison_output_dir"), session_dir=session_dir)
                prediction_dir = compare_root / chosen_variant
            else:
                prediction_dir = _resolve_artifact_path(payload.get("output_dir"), session_dir=session_dir)

            artifacts.append(
                Stage1FrameArtifact(
                    sample_name=str(payload["sample_name"]),
                    sequence_index=int(payload["sequence_index"]),
                    frame_id=int(payload["frame_id"]),
                    timestamp=float(payload["timestamp"]),
                    source_sample_dir=sample_dir,
                    source_prediction_dir=prediction_dir,
                    prediction_variant=chosen_variant,
                    stage1_status=status,
                )
            )

    artifacts.sort(key=lambda item: item.sequence_index)
    return artifacts


def _class_name(class_id: int, class_names: Sequence[str]) -> str:
    if 0 <= class_id < len(class_names):
        return str(class_names[class_id])
    return f"class_{class_id}"


def _class_group(class_name: str) -> str:
    return CLASS_GROUPS.get(class_name, "unknown")


def _velocity_from_box(box: Sequence[float]) -> List[float]:
    if len(box) >= 9:
        return [float(box[7]), float(box[8])]
    return [0.0, 0.0]


def _ego_state_from_meta(meta: Dict) -> EgoState:
    ego_velocity = meta["ego"]["velocity_carla"]
    speed_mps = float(math.sqrt(sum(float(component) ** 2 for component in ego_velocity)))
    world_from_ego = meta["ego"]["matrix_world_from_ego_bevfusion"]
    position_world = [
        float(world_from_ego[0][3]),
        float(world_from_ego[1][3]),
        float(world_from_ego[2][3]),
    ]
    return EgoState(
        frame_id=int(meta["frame_id"]),
        timestamp=float(meta["timestamp"]),
        sample_name=str(meta["sample_name"]),
        town=meta.get("town"),
        weather={key: float(value) for key, value in meta.get("weather", {}).items()},
        position_world=position_world,
        velocity_world=[
            float(ego_velocity[0]),
            float(-ego_velocity[1]),
            float(ego_velocity[2]),
        ],
        speed_mps=speed_mps,
        yaw_deg=float(meta["ego"]["transform_carla"]["rotation"]["yaw"]),
        route_progress_m=0.0,
        world_from_ego_bevfusion=[
            [float(value) for value in row] for row in world_from_ego
        ],
    )


def load_normalized_prediction(
    artifact: Stage1FrameArtifact,
    *,
    class_names: Sequence[str] | None = None,
    min_score: float = 0.2,
) -> NormalizedFramePrediction:
    meta = _load_json(artifact.source_sample_dir / "meta.json")
    predictions = _load_json(artifact.source_prediction_dir / "predictions.json")
    class_names = list(class_names or DEFAULT_OBJECT_CLASSES)

    boxes = np.asarray(predictions.get("boxes_3d", []), dtype=np.float32)
    scores = np.asarray(predictions.get("scores_3d", []), dtype=np.float32)
    labels = np.asarray(predictions.get("labels_3d", []), dtype=np.int64)
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError(
            f"Prediction tensor length mismatch for {artifact.sample_name}: "
            f"boxes={len(boxes)} scores={len(scores)} labels={len(labels)}"
        )

    ego_state = _ego_state_from_meta(meta)
    detections: List[NormalizedDetection] = []
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        score_f = float(score)
        if score_f < float(min_score):
            continue

        class_id = int(label)
        class_name = _class_name(class_id, class_names)
        class_group = _class_group(class_name)
        velocity_ego = _velocity_from_box(box)
        speed_mps = float(math.sqrt(float(velocity_ego[0]) ** 2 + float(velocity_ego[1]) ** 2))
        position_ego = [float(box[0]), float(box[1]), float(box[2])]
        distance_m = float(math.sqrt(float(box[0]) ** 2 + float(box[1]) ** 2))
        bearing_deg = float(math.degrees(math.atan2(float(box[1]), float(box[0]))))

        detections.append(
            NormalizedDetection(
                detection_id=f"{artifact.sample_name}_det_{idx:04d}",
                class_id=class_id,
                class_name=class_name,
                class_group=class_group,
                score=score_f,
                position_ego=position_ego,
                size_xyz=[float(box[3]), float(box[4]), float(box[5])] if len(box) >= 6 else [0.0, 0.0, 0.0],
                yaw_rad=float(box[6]) if len(box) >= 7 else 0.0,
                velocity_ego=velocity_ego,
                speed_mps=speed_mps,
                distance_m=distance_m,
                bearing_deg=bearing_deg,
                source_confidence=score_f,
                raw_box_9d=[float(value) for value in box.tolist()],
            )
        )

    detections.sort(key=lambda item: item.score, reverse=True)
    LOGGER.info(
        "Loaded normalized frame sample=%s variant=%s raw_boxes=%d filtered=%d min_score=%.2f",
        artifact.sample_name,
        artifact.prediction_variant,
        len(boxes),
        len(detections),
        min_score,
    )
    return NormalizedFramePrediction(
        sample_name=str(meta["sample_name"]),
        sequence_index=int(meta["sequence_index"]),
        frame_id=int(meta["frame_id"]),
        timestamp=float(meta["timestamp"]),
        prediction_variant=artifact.prediction_variant,
        source_sample_dir=str(artifact.source_sample_dir),
        source_prediction_dir=str(artifact.source_prediction_dir),
        ego=ego_state,
        detections=detections,
        class_names=list(class_names),
        raw_prediction_count=int(len(boxes)),
        filtered_prediction_count=int(len(detections)),
    )


def load_sequence_predictions(
    stage1_session_dir: str | Path,
    *,
    prediction_variant: str = "auto",
    class_names: Sequence[str] | None = None,
    min_score: float = 0.2,
) -> List[NormalizedFramePrediction]:
    artifacts = list_stage1_frame_artifacts(stage1_session_dir, prediction_variant=prediction_variant)
    return [
        load_normalized_prediction(artifact, class_names=class_names, min_score=min_score)
        for artifact in artifacts
    ]


def summarize_prediction_sequence(frames: Iterable[NormalizedFramePrediction]) -> Dict[str, float]:
    frames = list(frames)
    if not frames:
        return {
            "num_frames": 0,
            "avg_filtered_detections": 0.0,
            "max_filtered_detections": 0.0,
        }
    filtered_counts = [frame.filtered_prediction_count for frame in frames]
    return {
        "num_frames": float(len(frames)),
        "avg_filtered_detections": float(sum(filtered_counts) / len(filtered_counts)),
        "max_filtered_detections": float(max(filtered_counts)),
    }
