from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage3c_coverage.junction_execution_runner import run_junction_execution
from stage3c_coverage.left_positive_scenario_helper import probe_left_positive, spawn_left_positive
from stage3c_coverage.scenario_coverage_analyzer import analyze_stage3c_coverage
from stage3c_coverage.visualization_stage3c_coverage import save_stage3c_coverage_visualization


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3C execution coverage tooling.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze Stage 3C execution coverage from existing artifacts.")
    analyze_parser.add_argument("--stage3c-root", default=r"D:\Agent-AI\outputs\stage3c")
    analyze_parser.add_argument("--stage3b-junction-dir", default=r"D:\Agent-AI\outputs\stage3b\st3b_junction_straight_v1")
    analyze_parser.add_argument("--stage3a-junction-dir", default=r"D:\Agent-AI\outputs\stage3\live_watch_compare_bridge_thr010_stage3a")
    analyze_parser.add_argument("--output-dir", required=True)

    junction_parser = subparsers.add_parser("run-junction", help="Run Stage 3C closed-loop on a junction Stage 3B session.")
    junction_parser.add_argument("--stage3b-output-dir", required=True)
    junction_parser.add_argument("--output-dir", required=True)
    junction_parser.add_argument("--scenario-manifest", default=None)
    junction_parser.add_argument("--attach-to-actor-id", type=int, default=0)
    junction_parser.add_argument("--carla-host", default="127.0.0.1")
    junction_parser.add_argument("--carla-port", type=int, default=2000)
    junction_parser.add_argument("--carla-timeout-s", type=float, default=10.0)
    junction_parser.add_argument("--carla-pythonapi-root", default=r"D:\carla\PythonAPI")
    junction_parser.add_argument("--fixed-delta-seconds", type=float, default=0.10)
    junction_parser.add_argument("--warmup-ticks", type=int, default=3)
    junction_parser.add_argument("--post-settle-ticks", type=int, default=20)
    junction_parser.add_argument("--max-frames", type=int, default=0)
    junction_parser.add_argument("--prepare-offset-m", type=float, default=0.75)
    junction_parser.add_argument("--sampling-radius-m", type=float, default=2.0)
    junction_parser.add_argument("--lane-change-success-hold-ticks", type=int, default=4)
    junction_parser.add_argument("--max-lane-change-ticks", type=int, default=45)
    junction_parser.add_argument("--lane-change-min-lateral-shift-m", type=float, default=0.50)
    junction_parser.add_argument("--lane-change-min-lateral-shift-fraction", type=float, default=0.15)
    junction_parser.add_argument("--stop-hold-ticks", type=int, default=10)
    junction_parser.add_argument("--stop-full-stop-speed-threshold-mps", type=float, default=0.35)
    junction_parser.add_argument("--stop-near-distance-m", type=float, default=8.0)
    junction_parser.add_argument("--stop-hard-brake-distance-m", type=float, default=3.0)
    junction_parser.add_argument("--record-mp4", action="store_true")
    junction_parser.add_argument("--recording-path", default=None)
    junction_parser.add_argument("--recording-camera-mode", default="chase", choices=["chase", "hood", "topdown"])
    junction_parser.add_argument("--recording-width", type=int, default=1280)
    junction_parser.add_argument("--recording-height", type=int, default=720)
    junction_parser.add_argument("--recording-fov", type=float, default=90.0)
    junction_parser.add_argument("--recording-fps", type=float, default=0.0)
    junction_parser.add_argument("--recording-post-roll-ticks", type=int, default=0)
    junction_parser.add_argument("--recording-overlay", action="store_true")
    junction_parser.add_argument("--recording-no-overlay", dest="recording_overlay", action="store_false")
    junction_parser.set_defaults(recording_overlay=True)

    left_parser = subparsers.add_parser("prepare-left", help="Probe and optionally spawn a left-positive multi-lane scenario.")
    left_parser.add_argument("--probe-output", required=True)
    left_parser.add_argument("--output-manifest", required=True)
    left_parser.add_argument("--rank", type=int, default=0)
    left_parser.add_argument("--top-k", type=int, default=6)
    left_parser.add_argument("--host", default="127.0.0.1")
    left_parser.add_argument("--port", type=int, default=2000)
    left_parser.add_argument("--town", default="Carla/Maps/Town10HD_Opt")
    left_parser.add_argument("--sample-distance-m", type=float, default=5.0)
    left_parser.add_argument("--min-forward-length-m", type=float, default=80.0)
    left_parser.add_argument("--blocker-distance-m", type=float, default=24.0)
    left_parser.add_argument("--adjacent-distance-m", type=float, default=42.0)
    left_parser.add_argument("--no-spawn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "analyze":
        payload = analyze_stage3c_coverage(
            stage3c_root=args.stage3c_root,
            stage3b_junction_dir=args.stage3b_junction_dir,
            stage3a_junction_dir=args.stage3a_junction_dir,
        )
        output_dir = Path(args.output_dir)
        _write_json(output_dir / "stage3c_gap_analysis.json", payload["gap_analysis"])
        _write_json(output_dir / "execution_coverage_summary.json", payload["execution_coverage_summary"])
        _write_json(output_dir / "coverage_bundle.json", payload)
        for run_metrics in payload["run_inventory"]:
            run_output_dir = output_dir / "runs" / str(run_metrics["run_name"])
            _write_json(run_output_dir / "run_metrics.json", run_metrics)
            _write_jsonl(run_output_dir / "stop_events.jsonl", list(run_metrics.get("stop_events", [])))
            _write_jsonl(
                run_output_dir / "lane_change_events_enriched.jsonl",
                list(run_metrics.get("lane_change_events", [])),
            )
        save_stage3c_coverage_visualization(output_dir=output_dir / "visualization", coverage_summary=payload)
        print(json.dumps(payload["execution_coverage_summary"], indent=2))
        return

    if args.command == "run-junction":
        summary = run_junction_execution(
            stage3b_output_dir=args.stage3b_output_dir,
            output_dir=args.output_dir,
            scenario_manifest=args.scenario_manifest,
            attach_to_actor_id=args.attach_to_actor_id,
            carla_host=args.carla_host,
            carla_port=args.carla_port,
            carla_timeout_s=args.carla_timeout_s,
            carla_pythonapi_root=args.carla_pythonapi_root,
            fixed_delta_seconds=args.fixed_delta_seconds,
            warmup_ticks=args.warmup_ticks,
            post_settle_ticks=args.post_settle_ticks,
            max_frames=args.max_frames,
            prepare_offset_m=args.prepare_offset_m,
            sampling_radius_m=args.sampling_radius_m,
            lane_change_success_hold_ticks=args.lane_change_success_hold_ticks,
            max_lane_change_ticks=args.max_lane_change_ticks,
            lane_change_min_lateral_shift_m=args.lane_change_min_lateral_shift_m,
            lane_change_min_lateral_shift_fraction=args.lane_change_min_lateral_shift_fraction,
            stop_hold_ticks=args.stop_hold_ticks,
            stop_full_stop_speed_threshold_mps=args.stop_full_stop_speed_threshold_mps,
            stop_near_distance_m=args.stop_near_distance_m,
            stop_hard_brake_distance_m=args.stop_hard_brake_distance_m,
            record_mp4=args.record_mp4,
            recording_path=args.recording_path,
            recording_camera_mode=args.recording_camera_mode,
            recording_width=args.recording_width,
            recording_height=args.recording_height,
            recording_fov=args.recording_fov,
            recording_fps=args.recording_fps,
            recording_post_roll_ticks=args.recording_post_roll_ticks,
            recording_overlay=args.recording_overlay,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "prepare-left":
        candidates = probe_left_positive(
            output_json=args.probe_output,
            host=args.host,
            port=args.port,
            town=args.town,
            top_k=args.top_k,
            sample_distance_m=args.sample_distance_m,
            min_forward_length_m=args.min_forward_length_m,
        )
        payload = {
            "probe_output": str(args.probe_output),
            "num_candidates": len(candidates),
            "selected_rank": int(args.rank),
            "selected_candidate": None if not candidates else candidates[max(0, min(int(args.rank), len(candidates) - 1))],
        }
        if not args.no_spawn and candidates:
            manifest = spawn_left_positive(
                output_manifest=args.output_manifest,
                probe_json=args.probe_output,
                rank=args.rank,
                host=args.host,
                port=args.port,
                town=args.town,
                blocker_distance_m=args.blocker_distance_m,
                adjacent_distance_m=args.adjacent_distance_m,
            )
            payload["manifest"] = manifest
        _write_json(Path(args.output_manifest).with_suffix(".coverage_helper.json"), payload)
        print(json.dumps(payload, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
