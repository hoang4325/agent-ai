from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carla_bevfusion_stage1.adapter import BEVFusionSampleAdapter  # noqa: E402
from carla_bevfusion_stage1.bevfusion_runtime import build_bevfusion_model  # noqa: E402
from carla_bevfusion_stage1.visualization import (  # noqa: E402
    save_prediction_comparison_bundle,
    save_prediction_debug_bundle,
)

LOGGER = logging.getLogger("offline_infer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline BEVFusion inference on dumped CARLA samples.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--radar-bridge", choices=("minimal",), default="minimal")
    parser.add_argument("--radar-ablation", choices=("none", "zero_bev"), default="none")
    parser.add_argument("--compare-zero-radar", dest="compare_zero_radar", action="store_true")
    parser.add_argument("--no-compare-zero-radar", dest="compare_zero_radar", action="store_false")
    parser.set_defaults(compare_zero_radar=None)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def serialize_output(output: Dict[str, Any]) -> Dict[str, Any]:
    boxes = output.get("boxes_3d")
    boxes_payload = boxes.tensor.detach().cpu().numpy().tolist() if boxes is not None and len(boxes) else []
    return {
        "boxes_3d": boxes_payload,
        "scores_3d": output.get("scores_3d").detach().cpu().numpy().tolist()
        if "scores_3d" in output
        else [],
        "labels_3d": output.get("labels_3d").detach().cpu().numpy().tolist()
        if "labels_3d" in output
        else [],
    }


def summarize_output(output: Dict[str, Any], class_names: List[str], score_threshold: float) -> Dict[str, Any]:
    if "scores_3d" not in output or "labels_3d" not in output:
        return {
            "num_boxes_total": 0,
            "num_boxes_at_threshold": 0,
            "top_score": 0.0,
            "per_class_at_threshold": {},
        }

    scores = output["scores_3d"].detach().cpu().numpy()
    labels = output["labels_3d"].detach().cpu().numpy()
    at_threshold = scores >= score_threshold
    per_class: Dict[str, int] = {}
    for label in labels[at_threshold]:
        class_name = class_names[int(label)] if int(label) < len(class_names) else str(int(label))
        per_class[class_name] = per_class.get(class_name, 0) + 1

    return {
        "num_boxes_total": int(scores.shape[0]),
        "num_boxes_at_threshold": int(at_threshold.sum()),
        "top_score": float(scores.max()) if scores.size else 0.0,
        "per_class_at_threshold": per_class,
    }


def log_adapter_debug(mode_name: str, adapted, model, score_threshold: float) -> None:
    radar_debug = adapted.debug["radar_debug"]
    lidar_debug = adapted.debug["lidar_debug"]
    LOGGER.info(
        "[%s] camera_order=%s image_tensor_shape=%s",
        mode_name,
        adapted.debug["meta"]["camera_order_for_model"],
        adapted.debug["image_tensor_shape"],
    )
    LOGGER.info(
        "[%s] lidar_points=%d radar_raw=%d radar_in_bev_xy=%d radar_in_full_xyz=%d radar_final=%d",
        mode_name,
        lidar_debug["final_point_count"],
        radar_debug["raw_total"],
        radar_debug["in_bev_xy_total"],
        radar_debug["in_full_xyz_total"],
        radar_debug["final_point_count"],
    )
    LOGGER.info(
        "[%s] lidar_history=%s radar_sweeps_used=%d radar_bridge=%s radar_ablation=%s",
        mode_name,
        [item["sample_name"] for item in lidar_debug["history_used"]],
        radar_debug["radar_sweeps_used"],
        getattr(model, "_carla_radar_runtime", {}).get("radar_bridge"),
        getattr(model, "_carla_radar_ablation", "none"),
    )
    LOGGER.info(
        "[%s] radar weight slice=%s threshold=%.2f",
        mode_name,
        getattr(model, "_carla_radar_runtime", {}).get("radar_bridge_details", {}),
        score_threshold,
    )


def run_single_inference(
    *,
    model,
    adapter: BEVFusionSampleAdapter,
    sample_dir: Path,
    output_dir: Path,
    box_type_3d_cls,
    device: str,
    score_threshold: float,
    radar_mode: str,
    radar_ablation: str,
) -> Dict[str, Any]:
    LOGGER.info(
        "Starting inference sample=%s radar_mode=%s radar_ablation=%s output_dir=%s",
        sample_dir,
        radar_mode,
        radar_ablation,
        output_dir,
    )
    model.set_carla_radar_ablation(radar_ablation)
    LOGGER.info("[%s] Building adapted sample", radar_ablation)
    adapted = adapter.build_from_sample_dir(
        sample_dir,
        radar_mode=radar_mode,
        box_type_3d_cls=box_type_3d_cls,
        device=device,
    )
    log_adapter_debug(radar_ablation, adapted, model, score_threshold)

    LOGGER.info("[%s] Running model forward", radar_ablation)
    outputs = model(**adapted.model_inputs)
    LOGGER.info("[%s] Model forward complete", radar_ablation)
    if not isinstance(outputs, list) or not outputs:
        raise RuntimeError(f"Unexpected inference output: {type(outputs)}")
    output = outputs[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    adapted.debug["object_classes"] = list(getattr(model, "CLASSES", ()))
    save_prediction_debug_bundle(
        output_root=output_dir,
        debug=adapted.debug,
        output=output,
        score_threshold=score_threshold,
    )

    summary = summarize_output(output, adapted.debug["object_classes"], score_threshold)
    LOGGER.info(
        "[%s] detections total=%d at_threshold=%d top_score=%.4f per_class=%s",
        radar_ablation,
        summary["num_boxes_total"],
        summary["num_boxes_at_threshold"],
        summary["top_score"],
        summary["per_class_at_threshold"],
    )

    _write_json(output_dir / "predictions.json", serialize_output(output))
    _write_json(
        output_dir / "adapter_report.json",
        {
            "sample_name": adapted.debug["sample_name"],
            "radar_mode": radar_mode,
            "radar_ablation": radar_ablation,
            "runtime_info": getattr(model, "_carla_radar_runtime", {}),
            "lidar_debug": adapted.debug["lidar_debug"],
            "radar_debug": adapted.debug["radar_debug"],
        },
    )
    return {
        "adapted": adapted,
        "output": output,
        "summary": summary,
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    configure_logging()

    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    compare_zero_radar = (
        args.compare_zero_radar if args.compare_zero_radar is not None else args.radar_ablation == "none"
    )

    LOGGER.info("Building BEVFusion runtime model")
    model, cfg, box_type_3d_cls = build_bevfusion_model(
        repo_root=args.repo_root,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        radar_bridge=args.radar_bridge,
        radar_ablation=args.radar_ablation,
    )
    LOGGER.info("Creating sample adapter")
    adapter = BEVFusionSampleAdapter(cfg._cfg_dict.to_dict())

    if compare_zero_radar and args.radar_ablation == "none":
        LOGGER.info("Running baseline zero_bev pass before minimal radar bridge pass")
        baseline_result = run_single_inference(
            model=model,
            adapter=adapter,
            sample_dir=sample_dir,
            output_dir=output_dir / "baseline_zero_bev",
            box_type_3d_cls=box_type_3d_cls,
            device=args.device,
            score_threshold=args.score_threshold,
            radar_mode=args.radar_bridge,
            radar_ablation="zero_bev",
        )
        bridge_result = run_single_inference(
            model=model,
            adapter=adapter,
            sample_dir=sample_dir,
            output_dir=output_dir / "bridge_minimal",
            box_type_3d_cls=box_type_3d_cls,
            device=args.device,
            score_threshold=args.score_threshold,
            radar_mode=args.radar_bridge,
            radar_ablation="none",
        )

        comparison_payload = {
            "sample_name": bridge_result["adapted"].debug["sample_name"],
            "radar_bridge": args.radar_bridge,
            "baseline_zero_bev": baseline_result["summary"],
            "bridge_minimal": bridge_result["summary"],
            "runtime_info": getattr(model, "_carla_radar_runtime", {}),
            "radar_debug": bridge_result["adapted"].debug["radar_debug"],
            "lidar_debug": bridge_result["adapted"].debug["lidar_debug"],
        }
        _write_json(output_dir / "comparison.json", comparison_payload)
        save_prediction_comparison_bundle(
            output_root=output_dir,
            baseline_debug=baseline_result["adapted"].debug,
            baseline_output=baseline_result["output"],
            bridge_debug=bridge_result["adapted"].debug,
            bridge_output=bridge_result["output"],
            score_threshold=args.score_threshold,
        )
        LOGGER.info("Offline inference comparison complete. Outputs saved to %s", output_dir)
        return

    result = run_single_inference(
        model=model,
        adapter=adapter,
        sample_dir=sample_dir,
        output_dir=output_dir,
        box_type_3d_cls=box_type_3d_cls,
        device=args.device,
        score_threshold=args.score_threshold,
        radar_mode=args.radar_bridge,
        radar_ablation=args.radar_ablation,
    )
    LOGGER.info("Offline inference complete. Outputs saved to %s", result["output_dir"])


if __name__ == "__main__":
    main()
