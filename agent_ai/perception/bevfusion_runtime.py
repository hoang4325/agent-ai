from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from types import MethodType
from contextlib import contextmanager
from typing import Any, Dict, Tuple

import torch

from .config_loader import load_recursive_yaml_config
from .constants import (
    MINIMAL_RADAR_CHECKPOINT_COLUMN_INDICES,
    MINIMAL_RADAR_CHECKPOINT_COLUMN_NAMES,
    MINIMAL_RADAR_COLUMNS,
)

LOGGER = logging.getLogger(__name__)

RADAR_LINEAR_WEIGHT_KEY = "encoders.radar.backbone.pts_voxel_encoder.rfn_layers.0.linear.weight"
ALLOWED_RADAR_BRIDGES = {"minimal"}
ALLOWED_RADAR_ABLATIONS = {"none", "zero_bev"}


def load_runtime_config(config_path: str | Path) -> Dict[str, Any]:
    return load_recursive_yaml_config(config_path)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def _patch_cfg_for_minimal_radar_bridge(resolved_cfg: Dict[str, Any]) -> None:
    radar_encoder_cfg = resolved_cfg["model"]["encoders"]["radar"]["backbone"]["pts_voxel_encoder"]
    radar_encoder_cfg["in_channels"] = len(MINIMAL_RADAR_COLUMNS)


def _disable_pretrained_backbones_for_inference(resolved_cfg: Dict[str, Any]) -> None:
    """Avoid unnecessary remote downloads on inference-only runs.

    The task-specific checkpoint already contains trained weights for the
    camera backbone, so keeping backbone-level ``init_cfg=Pretrained`` only
    causes an avoidable download on first run.
    """
    model_cfg = resolved_cfg.get("model", {})
    encoders_cfg = model_cfg.get("encoders", {})
    camera_cfg = encoders_cfg.get("camera", {})
    backbone_cfg = camera_cfg.get("backbone", {})
    if isinstance(backbone_cfg, dict):
        backbone_cfg.pop("init_cfg", None)
        backbone_cfg["pretrained"] = None
        backbone_cfg["convert_weights"] = False


def _disable_training_only_head_config_for_inference(resolved_cfg: Dict[str, Any]) -> None:
    model_cfg = resolved_cfg.get("model", {})
    heads_cfg = model_cfg.get("heads", {})
    if isinstance(heads_cfg, dict):
        for head_name, head_cfg in heads_cfg.items():
            if isinstance(head_cfg, dict) and "train_cfg" in head_cfg:
                LOGGER.info("Disabling train_cfg for head '%s' during inference-only model build", head_name)
                head_cfg["train_cfg"] = None


@contextmanager
def _skip_bevfusion_init_weights():
    """Temporarily skip BEVFusion.init_weights during inference-only model build.

    The task checkpoint is loaded immediately after model construction, so
    paying the full random/pretrained init cost for the camera backbone is
    unnecessary and, on some environments, can stall for minutes.
    """
    try:
        from mmdet3d.models.fusion_models.bevfusion import BEVFusion as BEVFusionClass
    except Exception:
        yield
        return

    original_init_weights = BEVFusionClass.init_weights

    def _noop_init_weights(self) -> None:
        LOGGER.info("Skipping BEVFusion.init_weights during inference-only model construction")

    BEVFusionClass.init_weights = _noop_init_weights
    try:
        yield
    finally:
        BEVFusionClass.init_weights = original_init_weights


def _slice_minimal_radar_input_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    if RADAR_LINEAR_WEIGHT_KEY not in state_dict:
        raise KeyError(f"Missing checkpoint key required for radar bridge: {RADAR_LINEAR_WEIGHT_KEY}")

    original_weight = state_dict[RADAR_LINEAR_WEIGHT_KEY]
    expected_columns = max(MINIMAL_RADAR_CHECKPOINT_COLUMN_INDICES) + 1
    if original_weight.ndim != 2 or original_weight.shape[1] < expected_columns:
        raise ValueError(
            f"Unexpected radar input layer shape {tuple(original_weight.shape)}. "
            f"Need at least {expected_columns} columns to apply the minimal radar bridge."
        )

    keep_indices = list(MINIMAL_RADAR_CHECKPOINT_COLUMN_INDICES)
    sliced_weight = original_weight[:, keep_indices].contiguous()
    state_dict[RADAR_LINEAR_WEIGHT_KEY] = sliced_weight
    return {
        "bridge_strategy": "minimal_6d_runtime_patch",
        "kept_input_columns": list(MINIMAL_RADAR_COLUMNS),
        "kept_checkpoint_columns": list(MINIMAL_RADAR_CHECKPOINT_COLUMN_NAMES),
        "kept_checkpoint_indices": keep_indices,
        "original_shape": list(original_weight.shape),
        "patched_shape": list(sliced_weight.shape),
    }


def _install_radar_runtime_hooks(model, radar_ablation: str) -> None:
    if radar_ablation not in ALLOWED_RADAR_ABLATIONS:
        raise ValueError(f"Unsupported radar_ablation={radar_ablation}")

    original_extract_features = model.extract_features

    def extract_features_with_radar_ablation(self, x, sensor):
        # Radar ablation is now handled in forward_single_with_stage_logging
        # to ensure spatial dimensionality matches exactly without crashing the voxelizer.
        return original_extract_features(x, sensor)

    def set_carla_radar_ablation(self, mode: str) -> None:
        if mode not in ALLOWED_RADAR_ABLATIONS:
            raise ValueError(f"Unsupported radar ablation mode: {mode}")
        self._carla_radar_ablation = mode

    model.extract_features = MethodType(extract_features_with_radar_ablation, model)
    model.set_carla_radar_ablation = MethodType(set_carla_radar_ablation, model)
    model._carla_radar_ablation = radar_ablation


def _install_forward_stage_logging(model) -> None:
    original_forward_single = model.forward_single

    def forward_single_with_stage_logging(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        LOGGER.info("Forward start sample=%s sensor_order=%s", metas[0].get("sample_idx"), list(self.sensor_order))
        features = []
        ordered_sensors = []
        sensor_extract_order = self.sensor_order if self.training else list(reversed(self.sensor_order))
        for sensor in sensor_extract_order:
            LOGGER.info("Forward sensor stage start: %s", sensor)
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                if self.use_depth_loss:
                    feature = feature[0]
            elif sensor == "lidar":
                feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                if getattr(self, "_carla_radar_ablation", "none") == "zero_bev":
                    # Bypass mmDet3D voxelizer entirely to preventCUDA empty grid crashes.
                    # Generate a clean zeroed feature map for SEFuser.
                    b, h, w = 1, 180, 180 # Fallback spatial shape
                    for _f in features:
                        if isinstance(_f, torch.Tensor) and _f.ndim == 4:
                            b, _, h, w = _f.shape
                            break
                    device = radar[0].device if isinstance(radar, list) and radar else "cuda"
                    # In our config: camera=80, lidar=256, radar=64 channels.
                    feature = torch.zeros((b, 64, h, w), dtype=torch.float32, device=device)
                else:
                    feature = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            LOGGER.info("Forward sensor stage done: %s shape=%s", sensor, tuple(feature.shape))
            features.append(feature)
            ordered_sensors.append(sensor)

        if not self.training:
            features = features[::-1]
            ordered_sensors = ordered_sensors[::-1]

        LOGGER.info("Forward fuser stage start ordered_sensors=%s", ordered_sensors)
        if self.fuser is not None:
            self._assert_fuser_compatibility(features, ordered_sensors)
            if self.fuser.__class__.__name__ == "SEFuser":
                x = self.fuser(features, sensor_order=ordered_sensors)
            else:
                x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]
        LOGGER.info("Forward fuser stage done shape=%s", tuple(x.shape))

        LOGGER.info("Forward decoder backbone start")
        x = self.decoder["backbone"](x)
        LOGGER.info("Forward decoder backbone done")
        x = self.decoder["neck"](x)
        if not isinstance(x, torch.Tensor):
            x = x[0]
        LOGGER.info("Forward decoder neck done shape=%s", tuple(x.shape))

        outputs = [{}]
        for head_type, head in self.heads.items():
            LOGGER.info("Forward head start: %s", head_type)
            if head_type == "object":
                pred_dict = head(x, metas)
                bboxes = head.get_bboxes(pred_dict, metas)
                for k, (boxes, scores, labels) in enumerate(bboxes):
                    outputs[k].update(
                        {
                            "boxes_3d": boxes.to("cpu"),
                            "scores_3d": scores.cpu(),
                            "labels_3d": labels.cpu(),
                        }
                    )
            elif head_type == "map":
                outputs[0]["masks_bev"] = head(x)
            else:
                raise ValueError(f"unsupported head: {head_type}")
            LOGGER.info("Forward head done: %s", head_type)

        LOGGER.info("Forward complete")
        return outputs

    model.forward_single = MethodType(forward_single_with_stage_logging, model)


def _load_checkpoint_with_bridge(model, checkpoint_path: Path, radar_bridge: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    LOGGER.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected checkpoint state_dict to be a dict, got {type(state_dict)}")
    state_dict = _strip_module_prefix(state_dict)
    state_dict = copy.deepcopy(state_dict)

    runtime_info: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "radar_bridge": radar_bridge,
        "radar_ablation": "none",
    }
    if radar_bridge == "minimal":
        LOGGER.info("Applying minimal radar bridge checkpoint slice on %s", RADAR_LINEAR_WEIGHT_KEY)
        runtime_info["radar_bridge_details"] = _slice_minimal_radar_input_weights(state_dict)
    else:
        raise ValueError(f"Unsupported radar_bridge={radar_bridge}")

    LOGGER.info("Loading state_dict into model")
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint load produced unexpected key mismatches: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    return checkpoint, runtime_info


def build_bevfusion_model(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cuda",
    radar_bridge: str = "minimal",
    radar_ablation: str = "none",
):
    if radar_bridge not in ALLOWED_RADAR_BRIDGES:
        raise ValueError(f"Unsupported radar_bridge={radar_bridge}")
    if radar_ablation not in ALLOWED_RADAR_ABLATIONS:
        raise ValueError(f"Unsupported radar_ablation={radar_ablation}")

    repo_root = Path(repo_root).resolve()
    config_path = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()

    LOGGER.info(
        "Preparing BEVFusion runtime repo_root=%s config=%s checkpoint=%s device=%s radar_bridge=%s radar_ablation=%s",
        repo_root,
        config_path,
        checkpoint_path,
        device,
        radar_bridge,
        radar_ablation,
    )

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        LOGGER.info("Importing mmcv.Config")
        from mmcv import Config
        LOGGER.info("Imported mmcv.Config")
        LOGGER.info("Importing mmcv.runner.wrap_fp16_model")
        from mmcv.runner import wrap_fp16_model
        LOGGER.info("Imported mmcv.runner.wrap_fp16_model")
        LOGGER.info("Importing mmdet3d.models.build_model")
        from mmdet3d.models import build_model
        LOGGER.info("Imported mmdet3d.models.build_model")
        LOGGER.info("Importing LiDARInstance3DBoxes")
        from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
        LOGGER.info("Imported LiDARInstance3DBoxes")
    except Exception as exc:
        raise RuntimeError(
            "BEVFusion runtime dependencies are missing from the current Python environment.\n"
            f"Python executable: {sys.executable}\n"
            f"Import failure: {exc!r}\n"
            "This repo's installation docs are Linux-only, so the recommended runtime is WSL/Linux with a dedicated "
            "BEVFusion environment that contains torch, mmcv, mmdet and mmdet3d."
        ) from exc

    LOGGER.info("Loading resolved config")
    resolved_cfg = load_runtime_config(config_path)
    _disable_pretrained_backbones_for_inference(resolved_cfg)
    _disable_training_only_head_config_for_inference(resolved_cfg)
    if radar_bridge == "minimal":
        _patch_cfg_for_minimal_radar_bridge(resolved_cfg)

    cfg = Config(resolved_cfg, filename=str(config_path))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    LOGGER.info("Building BEVFusion model graph")
    with _skip_bevfusion_init_weights():
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    LOGGER.info("Model graph built")
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        LOGGER.info("Wrapping model for fp16 inference")
        wrap_fp16_model(model)

    LOGGER.info("Loading checkpoint weights")
    checkpoint, runtime_info = _load_checkpoint_with_bridge(model, checkpoint_path, radar_bridge)
    _install_radar_runtime_hooks(model, radar_ablation)
    _install_forward_stage_logging(model)
    runtime_info["radar_ablation"] = radar_ablation

    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = tuple(cfg.get("object_classes", ()))

    model._carla_radar_runtime = runtime_info
    LOGGER.info("Moving model to device %s", device)
    model.to(device)
    LOGGER.info("Switching model to eval mode")
    model.eval()
    LOGGER.info(
        "Loaded checkpoint %s onto %s with radar_bridge=%s radar_ablation=%s",
        checkpoint_path,
        device,
        radar_bridge,
        radar_ablation,
    )
    return model, cfg, LiDARInstance3DBoxes


@torch.no_grad()
def run_bevfusion_inference(model, model_inputs: Dict[str, Any]):
    return model(**model_inputs)
