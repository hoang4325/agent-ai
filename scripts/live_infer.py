from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carla_bevfusion_stage1.adapter import BEVFusionSampleAdapter
from carla_bevfusion_stage1.bevfusion_runtime import build_bevfusion_model
from carla_bevfusion_stage1.constants import (
    LIDAR_SENSOR_NAME,
    MODEL_CAMERA_ORDER,
    RADAR_SENSOR_ORDER,
)
from carla_bevfusion_stage1.visualization import (
    save_prediction_comparison_bundle,
    save_prediction_debug_bundle,
)

if TYPE_CHECKING:
    import carla

LOGGER = logging.getLogger("live_infer")
SAMPLE_NAME_PATTERN = re.compile(r"sample_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run stage-1 live inference. "
            "Default mode captures from CARLA then infers. "
            "Watch mode reads already-dumped samples via --samples-root."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples-root", default=None)
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--warmup-ticks", type=int, default=5)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--image-height", type=int, default=900)
    parser.add_argument("--camera-fov", type=float, default=70.0)
    parser.add_argument("--vehicle-filter", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--spawn-point-index", type=int, default=0)
    parser.add_argument("--town", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.5)
    parser.add_argument("--idle-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--adapter-lidar-sweeps-test", type=int, default=0)
    parser.add_argument("--adapter-radar-sweeps", type=int, default=0)
    parser.add_argument("--adapter-lidar-max-points", type=int, default=0)
    parser.add_argument("--adapter-radar-max-points", type=int, default=0)
    parser.add_argument("--radar-bridge", choices=("minimal",), default="minimal")
    parser.add_argument("--radar-ablation", choices=("zero_bev", "none"), default="zero_bev")
    parser.add_argument("--compare-zero-radar", dest="compare_zero_radar", action="store_true")
    parser.add_argument("--no-compare-zero-radar", dest="compare_zero_radar", action="store_false")
    parser.set_defaults(compare_zero_radar=False)
    parser.add_argument("--autopilot", dest="autopilot", action="store_true")
    parser.add_argument("--no-autopilot", dest="autopilot", action="store_false")
    parser.set_defaults(autopilot=True)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _import_carla():
    try:
        import carla
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Python module 'carla' is not available in this environment. "
            "Use --samples-root watch mode inside the BEVFusion container, "
            "or run capture mode in an environment that has a CARLA Python API "
            "matching the simulator version."
        ) from exc
    return carla


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _make_session_root(output_root: Path, session_name: str | None) -> Path:
    if session_name:
        session_root = output_root / session_name
        if session_root.exists():
            raise FileExistsError(f"Live session output already exists: {session_root}")
        return session_root

    base = datetime.now().strftime("live_%Y%m%d_%H%M%S")
    session_root = output_root / base
    suffix = 1
    while session_root.exists():
        session_root = output_root / f"{base}_{suffix:02d}"
        suffix += 1
    return session_root


def _sample_sequence_key(sample_dir: Path) -> int:
    match = SAMPLE_NAME_PATTERN.match(sample_dir.name)
    if match:
        return int(match.group(1))
    meta_path = sample_dir / "meta.json"
    if meta_path.exists():
        try:
            return int(_load_json(meta_path).get("sequence_index", -1))
        except Exception:
            return -1
    return -1


def _sample_dir_ready(sample_dir: Path) -> bool:
    complete_marker = sample_dir / "sample_complete.json"
    if complete_marker.exists():
        try:
            payload = _load_json(complete_marker)
            if not bool(payload.get("ready", False)):
                return False
        except Exception:
            return False
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        return False
    if not (sample_dir / "images").exists():
        return False
    if not (sample_dir / "lidar" / f"{LIDAR_SENSOR_NAME}.npy").exists():
        return False
    for camera_name in MODEL_CAMERA_ORDER:
        if not (sample_dir / "images" / f"{camera_name}.png").exists():
            return False
    for radar_name in RADAR_SENSOR_ORDER:
        if not (sample_dir / "radar" / f"{radar_name}.npy").exists():
            return False
    return True


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


def summarize_output(output: Dict[str, Any], class_names, score_threshold: float) -> Dict[str, Any]:
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


def _build_runtime(args: argparse.Namespace):
    build_start = time.perf_counter()
    model, cfg, box_type_3d_cls = build_bevfusion_model(
        repo_root=args.repo_root,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        radar_bridge=args.radar_bridge,
        radar_ablation=args.radar_ablation,
    )
    build_seconds = time.perf_counter() - build_start
    adapter_cfg = cfg._cfg_dict.to_dict()
    overrides = {}
    if args.adapter_lidar_sweeps_test > 0:
        adapter_cfg["lidar_sweeps_test"] = int(args.adapter_lidar_sweeps_test)
        overrides["lidar_sweeps_test"] = int(args.adapter_lidar_sweeps_test)
    if args.adapter_radar_sweeps > 0:
        adapter_cfg["radar_sweeps"] = int(args.adapter_radar_sweeps)
        overrides["radar_sweeps"] = int(args.adapter_radar_sweeps)
    if args.adapter_lidar_max_points > 0:
        adapter_cfg["lidar_max_points"] = int(args.adapter_lidar_max_points)
        overrides["lidar_max_points"] = int(args.adapter_lidar_max_points)
    if args.adapter_radar_max_points > 0:
        adapter_cfg["radar_max_points"] = int(args.adapter_radar_max_points)
        overrides["radar_max_points"] = int(args.adapter_radar_max_points)
    if overrides:
        LOGGER.info("Applying adapter overrides: %s", overrides)
    adapter = BEVFusionSampleAdapter(adapter_cfg)
    history_gate_template = _history_gate(adapter, sample_position=0)
    return model, cfg, box_type_3d_cls, adapter, build_seconds, history_gate_template


def spawn_ego(world, args: argparse.Namespace):
    blueprints = world.get_blueprint_library().filter(args.vehicle_filter)
    if not blueprints:
        raise ValueError(f"No vehicle blueprint matches filter '{args.vehicle_filter}'")
    blueprint = blueprints[0]
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("CARLA map does not expose any spawn points.")
    spawn_index = max(0, min(args.spawn_point_index, len(spawn_points) - 1))
    actor = world.try_spawn_actor(blueprint, spawn_points[spawn_index])
    if actor is None:
        raise RuntimeError(f"Failed to spawn ego vehicle at spawn point index {spawn_index}")
    return actor


def _history_gate(adapter: BEVFusionSampleAdapter, sample_position: int) -> Dict[str, Any]:
    required_prev_lidar = int(adapter.lidar_sweeps_test)
    required_prev_radar = max(0, int(adapter.radar_sweeps) - 1)
    required_prev = max(required_prev_lidar, required_prev_radar)
    return {
        "required_previous_samples": required_prev,
        "required_previous_lidar_samples": required_prev_lidar,
        "required_previous_radar_samples": required_prev_radar,
        "available_previous_samples": sample_position,
        "ready": sample_position >= required_prev,
    }


def _run_live_mode(
    *,
    model,
    adapted,
    output_dir: Path,
    radar_ablation: str,
    score_threshold: float,
    sample_name: str,
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    class_names = list(getattr(model, "CLASSES", ()))

    model.set_carla_radar_ablation(radar_ablation)
    infer_start = time.perf_counter()
    outputs = model(**adapted.model_inputs)
    timings["infer_seconds"] = time.perf_counter() - infer_start

    if not isinstance(outputs, list) or not outputs:
        raise RuntimeError(f"Unexpected inference output type: {type(outputs)}")
    output = outputs[0]

    adapted.debug["object_classes"] = class_names
    viz_start = time.perf_counter()
    save_prediction_debug_bundle(
        output_root=output_dir,
        debug=adapted.debug,
        output=output,
        score_threshold=score_threshold,
    )
    timings["visualize_seconds"] = time.perf_counter() - viz_start

    summary = summarize_output(output, class_names, score_threshold)
    _write_json(output_dir / "predictions.json", serialize_output(output))
    _write_json(
        output_dir / "adapter_report.json",
        {
            "sample_name": sample_name,
            "radar_mode": getattr(model, "_carla_radar_runtime", {}).get("radar_bridge"),
            "radar_ablation": radar_ablation,
            "runtime_info": getattr(model, "_carla_radar_runtime", {}),
            "lidar_debug": adapted.debug["lidar_debug"],
            "radar_debug": adapted.debug["radar_debug"],
            "timings": timings,
        },
    )
    LOGGER.info(
        "[%s] detections total=%d at_threshold=%d top_score=%.4f per_class=%s",
        radar_ablation,
        summary["num_boxes_total"],
        summary["num_boxes_at_threshold"],
        summary["top_score"],
        summary["per_class_at_threshold"],
    )
    return {
        "output": output,
        "summary": summary,
        "timings": timings,
        "output_dir": str(output_dir),
    }


def _process_sample_for_live_inference(
    *,
    model,
    adapter: BEVFusionSampleAdapter,
    box_type_3d_cls,
    sample_dir: Path,
    sample_position: int,
    inference_root: Path,
    summary_path: Path,
    metrics: Dict[str, int],
    args: argparse.Namespace,
    compare_zero_radar: bool,
    dump_seconds: Optional[float] = None,
) -> None:
    meta = _load_json(sample_dir / "meta.json")
    history_gate = _history_gate(adapter, sample_position)
    summary_base: Dict[str, Any] = {
        "sample_name": meta["sample_name"],
        "sequence_index": int(meta["sequence_index"]),
        "frame_id": int(meta["frame_id"]),
        "timestamp": float(meta["timestamp"]),
        "town": meta.get("town"),
        "history_gate": history_gate,
        "radar_bridge": args.radar_bridge,
        "source_sample_dir": str(sample_dir),
    }
    if dump_seconds is not None:
        summary_base["dump_seconds"] = dump_seconds

    if not history_gate["ready"]:
        metrics["history_skipped_samples"] += 1
        payload = {
            **summary_base,
            "status": "skipped_history_not_ready",
            "available_previous_samples": history_gate["available_previous_samples"],
            "required_previous_samples": history_gate["required_previous_samples"],
        }
        _append_jsonl(summary_path, payload)
        LOGGER.info(
            "Skipping inference for %s frame=%d until history is ready (%d/%d previous samples)",
            meta["sample_name"],
            meta["frame_id"],
            history_gate["available_previous_samples"],
            history_gate["required_previous_samples"],
        )
        return

    adapt_start = time.perf_counter()
    adapted = adapter.build_from_sample_dir(
        sample_dir,
        radar_mode=args.radar_bridge,
        box_type_3d_cls=box_type_3d_cls,
        device=args.device,
    )
    adapt_seconds = time.perf_counter() - adapt_start

    LOGGER.info(
        "Live adapter sample=%s frame=%d lidar_points=%d radar_raw=%d radar_final=%d image_tensor_shape=%s",
        meta["sample_name"],
        meta["frame_id"],
        adapted.debug["lidar_debug"]["final_point_count"],
        adapted.debug["radar_debug"]["raw_total"],
        adapted.debug["radar_debug"]["final_point_count"],
        adapted.debug["image_tensor_shape"],
    )

    sample_infer_root = inference_root / meta["sample_name"]
    if compare_zero_radar:
        baseline = _run_live_mode(
            model=model,
            adapted=adapted,
            output_dir=sample_infer_root / "baseline_zero_bev",
            radar_ablation="zero_bev",
            score_threshold=args.score_threshold,
            sample_name=meta["sample_name"],
        )
        bridge = _run_live_mode(
            model=model,
            adapted=adapted,
            output_dir=sample_infer_root / "bridge_minimal",
            radar_ablation="none",
            score_threshold=args.score_threshold,
            sample_name=meta["sample_name"],
        )
        metrics["inference_runs"] += 2

        comparison_payload = {
            "sample_name": meta["sample_name"],
            "frame_id": meta["frame_id"],
            "timestamp": meta["timestamp"],
            "radar_bridge": args.radar_bridge,
            "baseline_zero_bev": baseline["summary"],
            "bridge_minimal": bridge["summary"],
            "runtime_info": getattr(model, "_carla_radar_runtime", {}),
            "radar_debug": adapted.debug["radar_debug"],
            "lidar_debug": adapted.debug["lidar_debug"],
            "timings": {
                "adapt_seconds": adapt_seconds,
                "baseline_infer_seconds": baseline["timings"]["infer_seconds"],
                "bridge_infer_seconds": bridge["timings"]["infer_seconds"],
            },
        }
        if dump_seconds is not None:
            comparison_payload["timings"]["dump_seconds"] = dump_seconds
        _write_json(sample_infer_root / "comparison.json", comparison_payload)
        save_prediction_comparison_bundle(
            output_root=sample_infer_root,
            baseline_debug=adapted.debug,
            baseline_output=baseline["output"],
            bridge_debug=adapted.debug,
            bridge_output=bridge["output"],
            score_threshold=args.score_threshold,
        )
        _append_jsonl(
            summary_path,
            {
                **summary_base,
                "status": "inferred_compare",
                "adapt_seconds": adapt_seconds,
                "baseline_zero_bev": baseline["summary"],
                "bridge_minimal": bridge["summary"],
                "comparison_output_dir": str(sample_infer_root),
            },
        )
        LOGGER.info(
            "Live compare saved for %s frame=%d output=%s",
            meta["sample_name"],
            meta["frame_id"],
            sample_infer_root,
        )
        return

    result = _run_live_mode(
        model=model,
        adapted=adapted,
        output_dir=sample_infer_root,
        radar_ablation=args.radar_ablation,
        score_threshold=args.score_threshold,
        sample_name=meta["sample_name"],
    )
    metrics["inference_runs"] += 1
    _append_jsonl(
        summary_path,
        {
            **summary_base,
            "status": "inferred",
            "radar_ablation": args.radar_ablation,
            "adapt_seconds": adapt_seconds,
            "infer_seconds": result["timings"]["infer_seconds"],
            "visualize_seconds": result["timings"]["visualize_seconds"],
            "summary": result["summary"],
            "output_dir": result["output_dir"],
        },
    )
    LOGGER.info(
        "Live inference saved for %s frame=%d radar_ablation=%s output=%s",
        meta["sample_name"],
        meta["frame_id"],
        args.radar_ablation,
        sample_infer_root,
    )


def _run_watch_mode(args: argparse.Namespace) -> None:
    samples_root = Path(args.samples_root).resolve()
    if not samples_root.exists():
        raise FileNotFoundError(f"Watch samples root does not exist: {samples_root}")

    output_root = Path(args.output_root)
    session_root = _make_session_root(output_root, args.session_name)
    inference_root = session_root / "inference"
    summary_path = session_root / "live_summary.jsonl"
    inference_root.mkdir(parents=True, exist_ok=True)

    compare_zero_radar = bool(args.compare_zero_radar and args.radar_ablation == "none")
    model, cfg, box_type_3d_cls, adapter, build_seconds, history_gate_template = _build_runtime(args)
    del cfg

    metrics = {
        "observed_samples": 0,
        "history_skipped_samples": 0,
        "inference_runs": 0,
    }
    processed: Set[str] = set()
    idle_start = time.monotonic()

    _write_json(
        session_root / "session_manifest.json",
        {
            "session_root": str(session_root),
            "mode": "watch",
            "source_samples_root": str(samples_root),
            "inference_root": str(inference_root),
            "args": vars(args),
            "model_runtime": getattr(model, "_carla_radar_runtime", {}),
            "build_seconds": build_seconds,
            "history_gate": history_gate_template,
            "compare_zero_radar": compare_zero_radar,
        },
    )

    LOGGER.info(
        "Starting watch mode source_samples_root=%s output_session=%s target_samples=%d",
        samples_root,
        session_root,
        args.num_samples,
    )

    while metrics["observed_samples"] < args.num_samples:
        ready_samples = sorted(
            [path for path in samples_root.iterdir() if path.is_dir() and path.name.startswith("sample_") and _sample_dir_ready(path)],
            key=_sample_sequence_key,
        )
        new_samples = [path for path in ready_samples if path.name not in processed]

        if new_samples:
            idle_start = time.monotonic()

        for sample_dir in new_samples:
            sample_position = ready_samples.index(sample_dir)
            processed.add(sample_dir.name)
            metrics["observed_samples"] += 1
            _process_sample_for_live_inference(
                model=model,
                adapter=adapter,
                box_type_3d_cls=box_type_3d_cls,
                sample_dir=sample_dir,
                sample_position=sample_position,
                inference_root=inference_root,
                summary_path=summary_path,
                metrics=metrics,
                args=args,
                compare_zero_radar=compare_zero_radar,
            )
            if metrics["observed_samples"] >= args.num_samples:
                break

        if metrics["observed_samples"] >= args.num_samples:
            break

        if args.idle_timeout_seconds > 0 and (time.monotonic() - idle_start) >= args.idle_timeout_seconds:
            LOGGER.info(
                "Stopping watch mode after %.1fs idle timeout with %d/%d samples processed",
                args.idle_timeout_seconds,
                metrics["observed_samples"],
                args.num_samples,
            )
            break

        time.sleep(max(0.1, args.poll_interval_seconds))

    _write_json(
        session_root / "session_summary.json",
        {
            "session_root": str(session_root),
            "mode": "watch",
            "observed_samples": metrics["observed_samples"],
            "history_skipped_samples": metrics["history_skipped_samples"],
            "inference_runs": metrics["inference_runs"],
            "compare_zero_radar": compare_zero_radar,
        },
    )
    LOGGER.info(
        "Watch session complete session_root=%s observed=%d history_skipped=%d inference_runs=%d",
        session_root,
        metrics["observed_samples"],
        metrics["history_skipped_samples"],
        metrics["inference_runs"],
    )


def _run_capture_and_infer_mode(args: argparse.Namespace) -> None:
    carla = _import_carla()

    from carla_bevfusion_stage1.collector import CarlaSynchronousMode, FrameCollector
    from carla_bevfusion_stage1.dumper import SampleDumper
    from carla_bevfusion_stage1.rig import default_nuscenes_like_rig, destroy_actors, spawn_rig

    compare_zero_radar = bool(args.compare_zero_radar and args.radar_ablation == "none")
    output_root = Path(args.output_root)
    session_root = _make_session_root(output_root, args.session_name)
    samples_root = session_root / "samples"
    inference_root = session_root / "inference"
    summary_path = session_root / "live_summary.jsonl"
    samples_root.mkdir(parents=True, exist_ok=True)
    inference_root.mkdir(parents=True, exist_ok=True)

    model, cfg, box_type_3d_cls, adapter, build_seconds, history_gate_template = _build_runtime(args)
    del cfg

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.load_world(args.town) if args.town else client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)

    ego_vehicle = None
    sensor_actors = {}
    collector = None
    metrics = {
        "captured_samples": 0,
        "history_skipped_samples": 0,
        "inference_runs": 0,
    }

    try:
        ego_vehicle = spawn_ego(world, args)
        if args.autopilot:
            ego_vehicle.set_autopilot(True, args.tm_port)
        rig = default_nuscenes_like_rig(
            image_width=args.image_width,
            image_height=args.image_height,
            camera_fov=args.camera_fov,
            fixed_delta_seconds=args.fixed_delta_seconds,
        )
        sensor_actors = spawn_rig(world, ego_vehicle, rig)
        collector = FrameCollector(sensor_actors, timeout_seconds=args.timeout_seconds)
        collector.start()
        dumper = SampleDumper(output_root=samples_root, rig_preset=rig, sensor_actors=sensor_actors)

        session_manifest = {
            "session_root": str(session_root),
            "mode": "capture_and_infer",
            "samples_root": str(samples_root),
            "inference_root": str(inference_root),
            "args": vars(args),
            "model_runtime": getattr(model, "_carla_radar_runtime", {}),
            "build_seconds": build_seconds,
            "history_gate": history_gate_template,
            "compare_zero_radar": compare_zero_radar,
            "town": world.get_map().name,
            "rig_name": rig.name,
            "ego_actor_id": int(ego_vehicle.id),
        }
        _write_json(session_root / "session_manifest.json", session_manifest)

        previous_sample_name = None
        with CarlaSynchronousMode(
            world,
            fixed_delta_seconds=args.fixed_delta_seconds,
            traffic_manager=traffic_manager,
        ) as sync_mode:
            for _ in range(args.warmup_ticks):
                frame = sync_mode.tick()
                snapshot = world.get_snapshot()
                collector.wait_for_frame(frame, snapshot)

            for sample_index in range(args.num_samples):
                frame = sync_mode.tick()
                snapshot = world.get_snapshot()
                capture = collector.wait_for_frame(frame, snapshot)

                dump_start = time.perf_counter()
                dump_result = dumper.dump_capture(
                    capture,
                    ego_vehicle=ego_vehicle,
                    world=world,
                    sample_index=sample_index,
                    previous_sample_name=previous_sample_name,
                )
                dump_seconds = time.perf_counter() - dump_start
                previous_sample_name = dump_result.sample_dir.name
                metrics["captured_samples"] += 1

                _process_sample_for_live_inference(
                    model=model,
                    adapter=adapter,
                    box_type_3d_cls=box_type_3d_cls,
                    sample_dir=dump_result.sample_dir,
                    sample_position=sample_index,
                    inference_root=inference_root,
                    summary_path=summary_path,
                    metrics=metrics,
                    args=args,
                    compare_zero_radar=compare_zero_radar,
                    dump_seconds=dump_seconds,
                )

    finally:
        if collector is not None:
            collector.stop()
        if sensor_actors:
            destroy_actors(sensor_actors.values())
        if ego_vehicle is not None:
            ego_vehicle.destroy()

    _write_json(
        session_root / "session_summary.json",
        {
            "session_root": str(session_root),
            "mode": "capture_and_infer",
            "captured_samples": metrics["captured_samples"],
            "history_skipped_samples": metrics["history_skipped_samples"],
            "inference_runs": metrics["inference_runs"],
            "compare_zero_radar": compare_zero_radar,
        },
    )
    LOGGER.info(
        "Live session complete session_root=%s captured=%d history_skipped=%d inference_runs=%d",
        session_root,
        metrics["captured_samples"],
        metrics["history_skipped_samples"],
        metrics["inference_runs"],
    )


def main() -> None:
    args = parse_args()
    configure_logging()
    if args.samples_root:
        _run_watch_mode(args)
        return
    _run_capture_and_infer_mode(args)


if __name__ == "__main__":
    main()
