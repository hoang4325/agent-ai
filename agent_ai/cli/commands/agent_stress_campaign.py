from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
PROJECT_ROOT = REPO_ROOT


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic Stage10 stress scenarios with attached Agent compare."
    )
    parser.add_argument("--carla-root", required=True)
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--bev-repo", required=True)
    parser.add_argument("--bev-config", required=True)
    parser.add_argument("--bev-ckpt", required=True)
    parser.add_argument("--device", default="cuda",
                        help="Torch device passed to Stage10 BEVFusion bridge")
    parser.add_argument("--map", default="Town10HD_Opt")
    parser.add_argument("--cases", default="full",
                        help=(
                            "Comma-separated deterministic stress cases to run. "
                            "Supported aliases: full, lane_only, ab_blocked_clear, "
                            "ab_blocked_clear_rear, ab_blocked_gap"
                        ))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--run-tag", default="",
                        help="Optional tag inserted into run folder names, useful for A/B runs")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--delta-t", type=float, default=0.1,
                        help="Simulation step passed to Stage10 bridge")
    parser.add_argument("--rig-profile", default="default",
                        choices=["default", "low_memory_shadow"],
                        help="Sensor rig quality preset passed to Stage10 bridge")
    parser.add_argument("--image-width", type=int, default=1600,
                        help="Camera sensor width passed to Stage10 bridge")
    parser.add_argument("--image-height", type=int, default=900,
                        help="Camera sensor height passed to Stage10 bridge")
    parser.add_argument("--score-thresh", type=float, default=0.35,
                        help="Detection score threshold passed to Stage10 bridge")
    parser.add_argument("--adapter-lidar-max-points", type=int, default=0,
                        help="Cap live LiDAR points before BEVFusion inference; 0 keeps all points")
    parser.add_argument("--adapter-radar-max-points", type=int, default=2500,
                        help="Cap live radar points before BEVFusion inference")
    parser.add_argument("--route-distance-m", type=float, default=None)
    parser.add_argument("--agent-mode", default="compare", choices=["stub", "api", "compare"])
    parser.add_argument("--agent-control-mode", default="shadow", choices=["baseline", "shadow", "assist"])
    parser.add_argument("--agent-trigger-mode", default="risk_or_event",
                        choices=["every_frame", "risk_or_event"])
    parser.add_argument("--agent-compare-stride", type=int, default=50)
    parser.add_argument("--agent-risk-ttc-threshold", type=float, default=2.0)
    parser.add_argument("--agent-assist-min-confidence", type=float, default=0.50)
    parser.add_argument("--agent-max-requests-per-minute", type=float, default=30.0,
                        help="Wall-clock rate limit for real Agent API calls")
    parser.add_argument("--radar-ablation", default="zero_bev", choices=["none", "zero_bev"])
    parser.add_argument("--record-mp4", action="store_true",
                        help="Record each Stage10 stress run to MP4")
    parser.add_argument("--recording-camera-mode", choices=("chase", "hood", "topdown", "topdown_wide"), default="chase")
    parser.add_argument("--recording-width", type=int, default=1280)
    parser.add_argument("--recording-height", type=int, default=720)
    parser.add_argument("--recording-fov", type=float, default=100.0)
    parser.add_argument("--recording-fps", type=float, default=0.0)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs" / "stage10_stress_campaign"))
    parser.add_argument("--report-root", default=str(PROJECT_ROOT / "agent_ai" / "benchmark" / "reports" / "stage10_stress_campaign"))
    parser.add_argument("--ego-autopilot", action="store_true",
                        help="Enable CARLA Traffic Manager autopilot on the attached ego. Leave off for clean closed-loop A/B control.")
    parser.add_argument("--npc-handbrake", action="store_true", default=True)
    parser.add_argument("--blocker-distance-m", type=float, default=None)
    parser.add_argument("--adjacent-distance-m", type=float, default=None)
    parser.add_argument("--adjacent-position", choices=["ahead", "behind", "both", "none"], default=None,
                        help="Place adjacent-lane NPC(s) ahead, behind, or both around ego.")
    parser.add_argument("--adjacent-front-distance-m", type=float, default=None)
    parser.add_argument("--adjacent-rear-distance-m", type=float, default=None)
    parser.add_argument("--moving-adjacent-npcs", action="store_true",
                        help="Let adjacent-lane NPCs move with Traffic Manager autopilot.")
    parser.add_argument("--adjacent-speed-diff-percent", type=float, default=None,
                        help="Traffic Manager speed difference for moving adjacent NPCs; positive means slower.")
    parser.add_argument("--spawn-probe-top-k", type=int, default=8,
                        help="Number of corridor candidates to probe for each deterministic spawn case")
    parser.add_argument("--spawn-max-candidates", type=int, default=6,
                        help="Maximum number of probed candidates to try before giving up on a case")
    return parser.parse_args()


def _run_command(command: List[str], *, label: str, check: bool = True) -> subprocess.CompletedProcess[Any]:
    t0 = time.monotonic()
    print(f"[stress-campaign] {label}")
    print("  " + " ".join(command))
    result = subprocess.run(command, check=check, cwd=str(PROJECT_ROOT))
    print(f"[stress-campaign] {label} finished in {time.monotonic() - t0:.1f}s")
    return result


def _case_option(args: argparse.Namespace, spec: Dict[str, Any], key: str, default: Any) -> Any:
    cli_value = getattr(args, key, None)
    if cli_value is not None:
        return cli_value
    return spec.get(key, default)


def _case_float(args: argparse.Namespace, spec: Dict[str, Any], key: str, default: float) -> float:
    return float(_case_option(args, spec, key, default))


def _resolve_case_options(args: argparse.Namespace, spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "blocker_distance_m": _case_float(args, spec, "blocker_distance_m", 24.0),
        "adjacent_distance_m": _case_float(args, spec, "adjacent_distance_m", 42.0),
        "adjacent_position": str(_case_option(args, spec, "adjacent_position", "ahead")),
        "adjacent_front_distance_m": _case_float(args, spec, "adjacent_front_distance_m", 0.0),
        "adjacent_rear_distance_m": _case_float(args, spec, "adjacent_rear_distance_m", 0.0),
        "moving_adjacent_npcs": bool(args.moving_adjacent_npcs or spec.get("moving_adjacent_npcs", False)),
        "adjacent_speed_diff_percent": _case_float(args, spec, "adjacent_speed_diff_percent", -80.0),
        "route_distance_m": _case_float(args, spec, "route_distance_m", 120.0),
    }


def _build_summary_payload(
    *,
    args: argparse.Namespace,
    case_specs: List[Dict[str, Any]],
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": "stage10_agent_stress_campaign_v1",
        "generated_at_utc": _utc_now_iso(),
        "map": args.map,
        "cases": [spec["case_label"] for spec in case_specs],
        "repeats": int(args.repeats),
        "max_frames": int(args.max_frames),
        "agent_mode": args.agent_mode,
        "agent_control_mode": args.agent_control_mode,
        "ego_autopilot": bool(args.ego_autopilot),
        "runs": runs,
    }


def _persist_progress(
    *,
    report_root: Path,
    args: argparse.Namespace,
    case_specs: List[Dict[str, Any]],
    runs: List[Dict[str, Any]],
) -> Path:
    payload = _build_summary_payload(args=args, case_specs=case_specs, runs=runs)
    latest_path = report_root / "campaign_latest.json"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return latest_path


def _load_case_report_summary(stage10_log_dir: Path) -> Dict[str, Any]:
    driving_path = stage10_log_dir / "stage10_driving_metrics.json"
    evaluation_path = stage10_log_dir / "stage10_agent_live_evaluation.json"
    assist_path = stage10_log_dir / "stage10_agent_assist_evaluation.json"
    if not driving_path.exists():
        return {}

    driving = json.loads(driving_path.read_text(encoding="utf-8"))
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.exists()
        else {}
    )
    assist = (
        json.loads(assist_path.read_text(encoding="utf-8"))
        if assist_path.exists()
        else {}
    )

    low_ttc_analysis = evaluation.get("low_ttc_analysis") or {}
    assist_query_frames = int(assist.get("agent_query_frames") or 0)
    assist_sim_frames = int(assist.get("sim_frames") or 0)
    queried_frames = int(evaluation.get("agent_queried_frames") or assist_query_frames or 0)
    sim_frames = int(evaluation.get("sim_frames") or assist_sim_frames or 0)
    collision_count = int(driving.get("collision_count") or 0)
    query_ratio = round(queried_frames / max(sim_frames, 1), 4) if sim_frames else None
    safety_outcome = "Collision" if collision_count > 0 else "Collision-free"
    route = driving.get("route") or {}

    return {
        "table2": {
            "frames": sim_frames or int(driving.get("frames") or 0),
            "route_progress_m": route.get("route_progress_m"),
            "distance_traveled_m": route.get("distance_traveled_m"),
            "collision_count": collision_count,
            "low_ttc_frames": int(low_ttc_analysis.get("total_low_ttc_frames") or 0),
            "safety_outcome": safety_outcome,
        },
        "table3": {
            "agreement_rate": evaluation.get("agreement_rate"),
            "disagreement_rate": evaluation.get("disagreement_rate"),
            "useful_disagreement_count": int(evaluation.get("useful_disagreement_count") or 0),
            "agent_queried_frames": queried_frames,
            "sim_frames": sim_frames,
            "query_ratio": query_ratio,
            "agent_fallback_rate": evaluation.get("agent_fallback_rate"),
            "low_ttc_agent_cautious_rate": low_ttc_analysis.get("agent_cautious_rate"),
            "low_ttc_baseline_cautious_rate": low_ttc_analysis.get("baseline_cautious_rate"),
            "assist_applied_frames": assist.get("assist_applied_frames"),
            "assist_intervention_rate": assist.get("assist_intervention_rate"),
            "assist_agent_query_frames": assist_query_frames,
            "assist_agent_query_rate": assist.get("agent_query_rate"),
            "agent_decision_applied_frames": assist.get("agent_decision_applied_frames"),
            "agent_decision_intervention_rate": assist.get("agent_decision_intervention_rate"),
            "assist_hold_applied_frames": assist.get("assist_hold_applied_frames"),
            "post_lane_change_cruise_frames": assist.get("post_lane_change_cruise_frames"),
            "controller_only_applied_frames": assist.get("controller_only_applied_frames"),
        },
    }


def _spawn_with_probe_retry(
    *,
    python_exe: str,
    args: argparse.Namespace,
    spec: Dict[str, Any],
    run_id: str,
    manifest_path: Path,
) -> Dict[str, Any]:
    probe_path = manifest_path.with_suffix(".probe.json")
    if manifest_path.exists():
        manifest_path.unlink()
    if probe_path.exists():
        probe_path.unlink()

    probe_cmd = [
        python_exe,
        "-m", "agent_ai.cli", "multilane_scenario",
        "--host", str(args.carla_host),
        "--port", str(args.carla_port),
        "--tm-port", str(args.tm_port),
        "--town", f"Carla/Maps/{args.map}",
        "probe",
        "--top-k", str(int(args.spawn_probe_top_k)),
        "--adjacent-side", str(spec["adjacent_side"]),
        "--output-json", str(probe_path),
    ]
    _run_command(probe_cmd, label=f"probe {run_id}")

    candidates = json.loads(probe_path.read_text(encoding="utf-8"))
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"No spawn candidates found for {run_id}")

    case_options = _resolve_case_options(args, spec)
    max_candidates = max(1, min(int(args.spawn_max_candidates), len(candidates)))
    last_error: str | None = None
    for rank in range(max_candidates):
        candidate = dict(candidates[rank])
        if manifest_path.exists():
            manifest_path.unlink()
        spawn_cmd = [
            python_exe,
            "-m", "agent_ai.cli", "multilane_scenario",
            "--host", str(args.carla_host),
            "--port", str(args.carla_port),
            "--tm-port", str(args.tm_port),
            "--town", f"Carla/Maps/{args.map}",
            "spawn",
            "--adjacent-side", str(spec["adjacent_side"]),
            "--road-id", str(int(candidate["road_id"])),
            "--section-id", str(int(candidate["section_id"])),
            "--lane-id", str(int(candidate["lane_id"])),
            "--s", str(float(candidate["s"])),
            "--blocker-distance-m", str(float(case_options["blocker_distance_m"])),
            "--adjacent-distance-m", str(float(case_options["adjacent_distance_m"])),
            "--adjacent-position", str(case_options["adjacent_position"]),
            "--adjacent-front-distance-m", str(float(case_options["adjacent_front_distance_m"])),
            "--adjacent-rear-distance-m", str(float(case_options["adjacent_rear_distance_m"])),
            "--adjacent-speed-diff-percent", str(float(case_options["adjacent_speed_diff_percent"])),
            "--output-manifest", str(manifest_path),
            "--cleanup-scenario-actors",
        ]
        if "blocker_kind" in spec:
            spawn_cmd.extend(["--blocker-kind", str(spec["blocker_kind"])])
        if "pedestrian_start_side" in spec:
            spawn_cmd.extend(["--pedestrian-start-side", str(spec["pedestrian_start_side"])])
        if "pedestrian_cross_speed_mps" in spec:
            spawn_cmd.extend(["--pedestrian-cross-speed-mps", str(float(spec["pedestrian_cross_speed_mps"]))])
        if "pedestrian_cross_lateral_m" in spec:
            spawn_cmd.extend(["--pedestrian-cross-lateral-m", str(float(spec["pedestrian_cross_lateral_m"]))])
        if bool(case_options["moving_adjacent_npcs"]):
            spawn_cmd.append("--moving-adjacent-npcs")
        if bool(args.npc_handbrake):
            spawn_cmd.append("--npc-handbrake")

        print(
            f"[stress-campaign] trying spawn candidate {rank + 1}/{max_candidates} "
            f"for {run_id} road={candidate['road_id']} lane={candidate['lane_id']} s={float(candidate['s']):.2f}"
        )
        result = _run_command(spawn_cmd, label=f"spawn {run_id} candidate={rank + 1}", check=False)
        if result.returncode == 0 and manifest_path.exists():
            return {
                "candidate_rank": rank,
                "candidate": candidate,
                "probe_path": str(probe_path),
            }

        last_error = f"spawn returncode={result.returncode}"
        if manifest_path.exists():
            cleanup_cmd = [
                python_exe,
                "-m", "agent_ai.cli", "multilane_scenario",
                "--host", str(args.carla_host),
                "--port", str(args.carla_port),
                "destroy",
                "--manifest", str(manifest_path),
            ]
            _run_command(cleanup_cmd, label=f"cleanup failed spawn {run_id} candidate={rank + 1}", check=False)
            try:
                manifest_path.unlink()
            except OSError:
                pass
        print(f"[stress-campaign] candidate {rank + 1} failed for {run_id} -> {last_error}")
        time.sleep(1.0)

    raise RuntimeError(f"Failed to spawn {run_id} after {max_candidates} candidates. last_error={last_error}")


def _case_specs(case_names: List[str]) -> List[Dict[str, Any]]:
    supported: Dict[str, Dict[str, Any]] = {
        "right": {
            "adjacent_side": "right",
            "case_label": "multilane_right_nominal",
            "blocker_distance_m": 24.0,
            "adjacent_distance_m": 42.0,
            "route_distance_m": 120.0,
        },
        "left": {
            "adjacent_side": "left",
            "case_label": "multilane_left_nominal",
            "blocker_distance_m": 24.0,
            "adjacent_distance_m": 42.0,
            "route_distance_m": 120.0,
        },
        "unsafe_right": {
            "adjacent_side": "right",
            "case_label": "unsafe_lane_change_right",
            "blocker_distance_m": 18.0,
            "adjacent_distance_m": 12.0,
            "route_distance_m": 90.0,
        },
        "unsafe_left": {
            "adjacent_side": "left",
            "case_label": "unsafe_lane_change_left",
            "blocker_distance_m": 18.0,
            "adjacent_distance_m": 12.0,
            "route_distance_m": 90.0,
        },
        "blocked_clear_right": {
            "adjacent_side": "right",
            "case_label": "blocked_lane_clear_right",
            "blocker_distance_m": 10.0,
            "adjacent_distance_m": 60.0,
            "route_distance_m": 60.0,
        },
        "blocked_clear_right_rear": {
            "adjacent_side": "right",
            "case_label": "blocked_lane_clear_right_rear_npc",
            "blocker_distance_m": 10.0,
            "adjacent_distance_m": 35.0,
            "adjacent_position": "behind",
            "route_distance_m": 60.0,
        },
        "blocked_gap_right": {
            "adjacent_side": "right",
            "case_label": "blocked_lane_gap_right",
            "blocker_distance_m": 18.0,
            "adjacent_distance_m": 38.0,
            "adjacent_position": "both",
            "adjacent_front_distance_m": 38.0,
            "adjacent_rear_distance_m": 22.0,
            "moving_adjacent_npcs": True,
            "adjacent_speed_diff_percent": -80.0,
            "route_distance_m": 60.0,
        },
        "pedestrian_cutout_right": {
            "adjacent_side": "right",
            "case_label": "pedestrian_cutout_right",
            "blocker_kind": "pedestrian",
            "blocker_distance_m": 10.0,
            "adjacent_distance_m": 60.0,
            "adjacent_position": "none",
            "pedestrian_start_side": "left",
            "pedestrian_cross_speed_mps": 2.8,
            "pedestrian_cross_lateral_m": 5.0,
            "route_distance_m": 60.0,
        },
        "blocked_clear_left": {
            "adjacent_side": "left",
            "case_label": "blocked_lane_clear_left",
            "blocker_distance_m": 10.0,
            "adjacent_distance_m": 60.0,
            "route_distance_m": 60.0,
        },
        "blocked_clear_left_rear": {
            "adjacent_side": "left",
            "case_label": "blocked_lane_clear_left_rear_npc",
            "blocker_distance_m": 10.0,
            "adjacent_distance_m": 35.0,
            "adjacent_position": "behind",
            "route_distance_m": 60.0,
        },
        "blocked_gap_left": {
            "adjacent_side": "left",
            "case_label": "blocked_lane_gap_left",
            "blocker_distance_m": 18.0,
            "adjacent_distance_m": 38.0,
            "adjacent_position": "both",
            "adjacent_front_distance_m": 38.0,
            "adjacent_rear_distance_m": 22.0,
            "moving_adjacent_npcs": True,
            "adjacent_speed_diff_percent": -80.0,
            "route_distance_m": 60.0,
        },
        "stop_follow": {
            "adjacent_side": "right",
            "case_label": "stop_follow_ambiguity",
            "blocker_distance_m": 10.0,
            "adjacent_distance_m": 14.0,
            "route_distance_m": 60.0,
        },
        "emergency_brake": {
            "adjacent_side": "right",
            "case_label": "emergency_brake_conflict",
            "blocker_distance_m": 6.0,
            "adjacent_distance_m": 16.0,
            "route_distance_m": 45.0,
        },
    }
    aliases = {
        "full": ["right", "left", "unsafe_right", "unsafe_left", "stop_follow", "emergency_brake"],
        "lane_only": ["right", "left"],
        "ab_blocked_clear": ["blocked_clear_right", "blocked_clear_left"],
        "ab_blocked_clear_rear": ["blocked_clear_right_rear", "blocked_clear_left_rear"],
        "ab_blocked_gap": ["blocked_gap_right", "blocked_gap_left"],
        "pedestrian_cutout": ["pedestrian_cutout_right"],
    }

    expanded: List[str] = []
    for case_name in case_names:
        key = case_name.strip().lower()
        if not key:
            continue
        expanded.extend(aliases.get(key, [key]))

    specs: List[Dict[str, Any]] = []
    for key in expanded:
        if key not in supported:
            labels = ", ".join(sorted({*supported.keys(), *aliases.keys()}))
            raise ValueError(f"Unsupported stress case '{key}'. Supported: {labels}")
        specs.append(dict(supported[key]))
    if not specs:
        raise ValueError("No stress cases selected.")
    return specs


def main() -> int:
    args = _parse_args()
    python_exe = sys.executable
    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    case_specs = _case_specs(str(args.cases).split(","))
    total_runs = len(case_specs) * max(1, int(args.repeats))

    run_counter = 0
    for repeat_idx in range(1, int(args.repeats) + 1):
        for spec in case_specs:
            run_counter += 1
            case_label = spec["case_label"]
            tag_part = f"_{str(args.run_tag).strip()}" if str(args.run_tag).strip() else ""
            run_id = f"{case_label}{tag_part}_run{repeat_idx}"
            manifest_path = output_root / f"{run_id}_manifest.json"
            stage10_output_root = output_root / run_id
            stage10_log_dir = report_root / run_id
            case_options = _resolve_case_options(args, spec)

            stage10_cmd = [
                python_exe,
                "-m", "agent_ai.cli", "live_authority_bridge",
                "--carla-root", str(args.carla_root),
                "--carla-host", str(args.carla_host),
                "--carla-port", str(args.carla_port),
                "--tm-port", str(args.tm_port),
                "--bev-repo", str(args.bev_repo),
                "--bev-config", str(args.bev_config),
                "--bev-ckpt", str(args.bev_ckpt),
                "--device", str(args.device),
                "--output-root", str(stage10_output_root),
                "--log-dir", str(stage10_log_dir),
                "--scenario-manifest", str(manifest_path),
                "--map", str(args.map),
                "--route-distance-m", str(float(case_options["route_distance_m"])),
                "--max-frames", str(args.max_frames),
                "--delta-t", str(args.delta_t),
                "--rig-profile", str(args.rig_profile),
                "--image-width", str(args.image_width),
                "--image-height", str(args.image_height),
                "--score-thresh", str(args.score_thresh),
                "--adapter-lidar-max-points", str(args.adapter_lidar_max_points),
                "--adapter-radar-max-points", str(args.adapter_radar_max_points),
                "--agent-mode", str(args.agent_mode),
                "--agent-control-mode", str(args.agent_control_mode),
                "--agent-trigger-mode", str(args.agent_trigger_mode),
                "--agent-compare-stride", str(args.agent_compare_stride),
                "--agent-risk-ttc-threshold", str(args.agent_risk_ttc_threshold),
                "--agent-assist-min-confidence", str(args.agent_assist_min_confidence),
                "--agent-max-requests-per-minute", str(args.agent_max_requests_per_minute),
                "--radar-ablation", str(args.radar_ablation),
            ]
            if bool(args.record_mp4):
                stage10_cmd.extend([
                    "--record-mp4",
                    "--recording-camera-mode", str(args.recording_camera_mode),
                    "--recording-width", str(args.recording_width),
                    "--recording-height", str(args.recording_height),
                    "--recording-fov", str(args.recording_fov),
                    "--recording-fps", str(args.recording_fps),
                ])
            if bool(args.ego_autopilot):
                stage10_cmd.append("--attach-autopilot")

            destroy_cmd = [
                python_exe,
                "-m", "agent_ai.cli", "multilane_scenario",
                "--host", str(args.carla_host),
                "--port", str(args.carla_port),
                "destroy",
                "--manifest", str(manifest_path),
            ]

            run_record: Dict[str, Any] = {
                "run_id": run_id,
                "case_label": case_label,
                "adjacent_side": spec["adjacent_side"],
                "repeat_index": repeat_idx,
                "manifest_path": str(manifest_path),
                "stage10_log_dir": str(stage10_log_dir),
                "stage10_output_root": str(stage10_output_root),
                "blocker_kind": str(spec.get("blocker_kind", "vehicle")),
                "blocker_distance_m": float(case_options["blocker_distance_m"]),
                "adjacent_distance_m": float(case_options["adjacent_distance_m"]),
                "adjacent_position": str(case_options["adjacent_position"]),
                "adjacent_front_distance_m": float(case_options["adjacent_front_distance_m"]),
                "adjacent_rear_distance_m": float(case_options["adjacent_rear_distance_m"]),
                "moving_adjacent_npcs": bool(case_options["moving_adjacent_npcs"]),
                "adjacent_speed_diff_percent": float(case_options["adjacent_speed_diff_percent"]),
                "route_distance_m": float(case_options["route_distance_m"]),
                "status": "pending",
                "started_at_utc": _utc_now_iso(),
            }
            runs.append(run_record)
            latest_path = _persist_progress(
                report_root=report_root,
                args=args,
                case_specs=case_specs,
                runs=runs,
            )

            print(
                f"[stress-campaign] progress {run_counter}/{total_runs} "
                f"repeat={repeat_idx}/{int(args.repeats)} case={case_label}"
            )
            print(f"[stress-campaign] latest summary -> {latest_path}")

            try:
                spawn_meta = _spawn_with_probe_retry(
                    python_exe=python_exe,
                    args=args,
                    spec=spec,
                    run_id=run_id,
                    manifest_path=manifest_path,
                )
                run_record["spawn_probe"] = spawn_meta
                _run_command(stage10_cmd, label=f"stage10 {run_id}")
                run_record["report_summary"] = _load_case_report_summary(stage10_log_dir)
                run_record["status"] = "completed"
            except Exception as exc:
                run_record["status"] = "failed"
                if isinstance(exc, subprocess.CalledProcessError):
                    run_record["failed_command"] = exc.cmd
                    run_record["exit_code"] = int(exc.returncode)
                run_record["error"] = str(exc)
            finally:
                try:
                    if manifest_path.exists():
                        _run_command(destroy_cmd, label=f"destroy {run_id}")
                except subprocess.CalledProcessError as exc:
                    run_record["destroy_status"] = f"failed:{exc.returncode}"
                else:
                    run_record["destroy_status"] = "completed"
                run_record["finished_at_utc"] = _utc_now_iso()
                _persist_progress(
                    report_root=report_root,
                    args=args,
                    case_specs=case_specs,
                    runs=runs,
                )

            if run_record["status"] != "completed":
                break
        else:
            continue
        break

    summary = _build_summary_payload(args=args, case_specs=case_specs, runs=runs)
    summary_path = report_root / f"campaign_summary_{int(time.time())}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[stress-campaign] summary -> {summary_path}")

    failed = [run for run in runs if run.get("status") != "completed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
