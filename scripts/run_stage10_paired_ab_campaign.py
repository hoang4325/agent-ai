from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired Stage10 baseline/async-Agent campaigns with identical seeds."
    )
    parser.add_argument("--carla-root", required=True)
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--bev-repo", required=True)
    parser.add_argument("--bev-config", required=True)
    parser.add_argument("--bev-ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--map", default="Town10HD_Opt")
    parser.add_argument("--cases", default="ab_blocked_clear")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--rig-profile", choices=["default", "low_memory_shadow"], default="default")
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--image-height", type=int, default=900)
    parser.add_argument("--score-thresh", type=float, default=0.35)
    parser.add_argument("--adapter-lidar-max-points", type=int, default=0)
    parser.add_argument("--adapter-radar-max-points", type=int, default=2500)
    parser.add_argument("--agent-trigger-mode", choices=["every_frame", "risk_or_event"], default="risk_or_event")
    parser.add_argument("--agent-compare-stride", type=int, default=10)
    parser.add_argument("--agent-risk-ttc-threshold", type=float, default=2.0)
    parser.add_argument("--agent-assist-min-confidence", type=float, default=0.70)
    parser.add_argument("--agent-max-requests-per-minute", type=float, default=30.0)
    parser.add_argument("--agent-max-requests-per-episode", type=int, default=2)
    parser.add_argument("--agent-api-timeout-s", type=float, default=15.0)
    parser.add_argument("--agent-api-max-retries", type=int, default=0)
    parser.add_argument("--agent-response-max-age-s", type=float, default=10.0)
    parser.add_argument("--agent-retry-cooldown-s", type=float, default=2.0)
    parser.add_argument("--agent-maneuver-timeout-s", type=float, default=15.0)
    parser.add_argument("--agent-lane-stable-frames", type=int, default=5)
    parser.add_argument("--agent-lane-center-tolerance-m", type=float, default=0.60)
    parser.add_argument("--agent-lane-heading-tolerance-rad", type=float, default=0.20)
    parser.add_argument("--agent-target-corridor-half-width-m", type=float, default=1.60)
    parser.add_argument("--agent-target-rear-clearance-m", type=float, default=8.0)
    parser.add_argument("--agent-emergency-ttc-floor-s", type=float, default=0.75)
    parser.add_argument("--radar-ablation", choices=["none", "zero_bev"], default="none")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "stage10_paired_ab"),
    )
    parser.add_argument(
        "--report-root",
        default=str(PROJECT_ROOT / "benchmark" / "reports" / "stage10_paired_ab"),
    )
    parser.add_argument("--record-mp4", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _common_campaign_args(args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_stage10_agent_stress_campaign.py"),
        "--carla-root", str(args.carla_root),
        "--carla-host", str(args.carla_host),
        "--carla-port", str(args.carla_port),
        "--tm-port", str(args.tm_port),
        "--bev-repo", str(args.bev_repo),
        "--bev-config", str(args.bev_config),
        "--bev-ckpt", str(args.bev_ckpt),
        "--device", str(args.device),
        "--map", str(args.map),
        "--cases", str(args.cases),
        "--seeds", str(args.seeds),
        "--max-frames", str(args.max_frames),
        "--delta-t", str(args.delta_t),
        "--rig-profile", str(args.rig_profile),
        "--image-width", str(args.image_width),
        "--image-height", str(args.image_height),
        "--score-thresh", str(args.score_thresh),
        "--adapter-lidar-max-points", str(args.adapter_lidar_max_points),
        "--adapter-radar-max-points", str(args.adapter_radar_max_points),
        "--agent-trigger-mode", str(args.agent_trigger_mode),
        "--agent-compare-stride", str(args.agent_compare_stride),
        "--agent-risk-ttc-threshold", str(args.agent_risk_ttc_threshold),
        "--agent-assist-min-confidence", str(args.agent_assist_min_confidence),
        "--agent-max-requests-per-minute", str(args.agent_max_requests_per_minute),
        "--agent-max-requests-per-episode", str(args.agent_max_requests_per_episode),
        "--agent-api-timeout-s", str(args.agent_api_timeout_s),
        "--agent-api-max-retries", str(args.agent_api_max_retries),
        "--agent-response-max-age-s", str(args.agent_response_max_age_s),
        "--agent-retry-cooldown-s", str(args.agent_retry_cooldown_s),
        "--agent-maneuver-timeout-s", str(args.agent_maneuver_timeout_s),
        "--agent-lane-stable-frames", str(args.agent_lane_stable_frames),
        "--agent-lane-center-tolerance-m", str(args.agent_lane_center_tolerance_m),
        "--agent-lane-heading-tolerance-rad", str(args.agent_lane_heading_tolerance_rad),
        "--agent-target-corridor-half-width-m", str(args.agent_target_corridor_half_width_m),
        "--agent-target-rear-clearance-m", str(args.agent_target_rear_clearance_m),
        "--agent-emergency-ttc-floor-s", str(args.agent_emergency_ttc_floor_s),
        "--radar-ablation", str(args.radar_ablation),
        "--output-root", str(args.output_root),
        "--report-root", str(args.report_root),
    ]
    if args.record_mp4:
        command.append("--record-mp4")
    return command


def _run(command: List[str], *, dry_run: bool) -> None:
    print("[paired-ab] " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))


def _geometry_signature(manifest: Dict[str, Any]) -> Dict[str, Any]:
    corridor = manifest.get("corridor") or {}
    return {
        "town": manifest.get("town"),
        "random_seed": manifest.get("random_seed"),
        "adjacent_side": manifest.get("adjacent_side"),
        "blocker_kind": manifest.get("blocker_kind"),
        "corridor": {
            "road_id": corridor.get("road_id"),
            "section_id": corridor.get("section_id"),
            "lane_id": corridor.get("lane_id"),
            "s": corridor.get("s"),
            "adjacent_lane_id": corridor.get("adjacent_lane_id"),
            "ego_transform": corridor.get("ego_transform"),
        },
        "placements": manifest.get("placements"),
    }


def _manifest_pairs(output_root: Path) -> Iterable[tuple[Path, Path]]:
    for baseline in sorted(output_root.glob("*_baseline_seed*_run*_manifest.json")):
        assist_name = baseline.name.replace("_baseline_seed", "_assist_seed", 1)
        yield baseline, output_root / assist_name


def _verify_paired_manifests(output_root: Path) -> Dict[str, Any]:
    checked = 0
    mismatches: List[Dict[str, Any]] = []
    for baseline_path, assist_path in _manifest_pairs(output_root):
        checked += 1
        if not assist_path.exists():
            mismatches.append(
                {"baseline": str(baseline_path), "assist": str(assist_path), "reason": "missing_assist"}
            )
            continue
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        assist = json.loads(assist_path.read_text(encoding="utf-8"))
        if _geometry_signature(baseline) != _geometry_signature(assist):
            mismatches.append(
                {"baseline": str(baseline_path), "assist": str(assist_path), "reason": "geometry_mismatch"}
            )
    report = {"checked_pairs": checked, "mismatches": mismatches, "passed": checked > 0 and not mismatches}
    if checked == 0:
        raise RuntimeError(f"No paired manifests found under {output_root}")
    if mismatches:
        raise RuntimeError(f"Paired manifest verification failed: {mismatches}")
    return report


def main() -> int:
    args = _parse_args()
    if int(args.max_frames) <= 0:
        raise ValueError("--max-frames must be positive")
    if float(args.delta_t) <= 0.0:
        raise ValueError("--delta-t must be positive")
    if float(args.agent_api_timeout_s) <= 0.0:
        raise ValueError("--agent-api-timeout-s must be positive")
    if int(args.agent_api_max_retries) < 0:
        raise ValueError("--agent-api-max-retries must be non-negative")
    if float(args.agent_response_max_age_s) <= 0.0:
        raise ValueError("--agent-response-max-age-s must be positive")
    if int(args.agent_max_requests_per_episode) < 0:
        raise ValueError("--agent-max-requests-per-episode must be non-negative")
    if float(args.agent_retry_cooldown_s) < 0.0:
        raise ValueError("--agent-retry-cooldown-s must be non-negative")
    if float(args.agent_maneuver_timeout_s) <= 0.0:
        raise ValueError("--agent-maneuver-timeout-s must be positive")
    if int(args.agent_lane_stable_frames) <= 0:
        raise ValueError("--agent-lane-stable-frames must be positive")
    if float(args.agent_lane_center_tolerance_m) < 0.0:
        raise ValueError("--agent-lane-center-tolerance-m must be non-negative")
    if float(args.agent_lane_heading_tolerance_rad) < 0.0:
        raise ValueError("--agent-lane-heading-tolerance-rad must be non-negative")

    common = _common_campaign_args(args)
    baseline_command = common + [
        "--run-tag", "baseline",
        "--agent-mode", "stub",
        "--agent-control-mode", "baseline",
    ]
    assist_command = common + [
        "--run-tag", "assist",
        "--agent-mode", "api",
        "--agent-control-mode", "assist",
    ]
    _run(baseline_command, dry_run=bool(args.dry_run))
    _run(assist_command, dry_run=bool(args.dry_run))

    if args.dry_run:
        return 0

    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    verification = _verify_paired_manifests(output_root)
    verification_path = report_root / "paired_manifest_verification.json"
    verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")

    summary_path = report_root / "paired_ab_summary.json"
    summary_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "summarize_stage10_stress_tables.py"),
        "--report-root", str(report_root),
        "--summary-json", str(summary_path),
    ]
    _run(summary_command, dry_run=False)
    print(f"[paired-ab] manifest verification -> {verification_path}", flush=True)
    print(f"[paired-ab] paired summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
