from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from stage3c.local_planner_bridge import ensure_carla_pythonapi


def _mp4_recording_disabled() -> bool:
    """Temporary safety switch to disable MP4 recording in test runs.

    Set AGENTAI_DISABLE_MP4=0 to re-enable recording.
    """
    return str(os.getenv("AGENTAI_DISABLE_MP4", "1")).strip().lower() not in {"0", "false", "no", "off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 4 online integrated loop: live CARLA sensors -> Stage1 -> Stage2 -> Stage3A -> Stage3B -> Stage3C."
    )
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--carla-timeout-s", type=float, default=10.0)
    parser.add_argument("--carla-ready-attempts", type=int, default=6)
    parser.add_argument("--carla-ready-retry-seconds", type=float, default=2.0)
    parser.add_argument("--carla-pythonapi-root", default=r"D:\carla\PythonAPI")
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--town", default=None)
    parser.add_argument("--load-town-if-needed", action="store_true")
    parser.add_argument("--scenario-manifest", default=None)
    parser.add_argument("--attach-to-actor-id", type=int, default=0)
    parser.add_argument("--spawn-ego-if-missing", action="store_true")
    parser.add_argument("--ego-vehicle-filter", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--ego-spawn-point-index", type=int, default=0)

    parser.add_argument("--num-ticks", type=int, default=60)
    parser.add_argument("--warmup-ticks", type=int, default=8)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.10)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    parser.add_argument("--max-frame-retries", type=int, default=8)
    parser.add_argument("--stage1-drain-timeout-seconds", type=float, default=4.0)

    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--camera-fov", type=float, default=70.0)
    parser.add_argument("--sensor-rig-profile", choices=("default", "low_memory_shadow"), default="default")

    parser.add_argument("--stage1-launch-mode", choices=("docker_watch", "external_watch"), default="docker_watch")
    parser.add_argument("--stage1-session-root", default=None)
    parser.add_argument("--stage1-session-name", default="stage4_online_stage1")
    parser.add_argument("--stage1-num-samples", type=int, default=0)
    parser.add_argument("--stage1-poll-interval-seconds", type=float, default=0.25)
    parser.add_argument("--stage1-idle-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--external-watch-start-sequence-index", type=int, default=-1)
    parser.add_argument("--external-watch-start-sample-name", default=None)
    parser.add_argument("--external-watch-end-sequence-index", type=int, default=-1)
    parser.add_argument("--external-watch-end-sample-name", default=None)
    parser.add_argument("--external-watch-max-payloads-per-poll", type=int, default=0)
    parser.add_argument("--stage1-container-name", default="bevfusion-cu128")
    parser.add_argument("--stage1-container-agent-root", default="/workspace/Agent-AI")
    parser.add_argument("--stage1-container-repo-root", default="/workspace/Bevfusion-Z")
    parser.add_argument(
        "--stage1-container-config",
        default="/workspace/Bevfusion-Z/configs/nuscenes/det/transfusion/secfpn/camera+lidar+radar/swint_v0p075/sefuser_radarbev.yaml",
    )
    parser.add_argument("--stage1-container-checkpoint", default="/workspace/Data-Train/epoch_8_3sensor.pth")
    parser.add_argument("--stage1-device", default="cuda")
    parser.add_argument("--prediction-variant", choices=("auto", "baseline_zero_bev", "bridge_minimal"), default="auto")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--radar-bridge", choices=("minimal",), default="minimal")
    parser.add_argument("--radar-ablation", choices=("zero_bev", "none"), default="zero_bev")
    parser.add_argument("--compare-zero-radar", action="store_true")
    parser.add_argument("--adapter-lidar-sweeps-test", type=int, default=3)
    parser.add_argument("--adapter-radar-sweeps", type=int, default=3)
    parser.add_argument("--adapter-lidar-max-points", type=int, default=120000)
    parser.add_argument("--adapter-radar-max-points", type=int, default=1200)

    parser.add_argument("--max-missed-frames", type=int, default=2)
    parser.add_argument("--vehicle-match-distance-m", type=float, default=5.0)
    parser.add_argument("--vru-match-distance-m", type=float, default=3.0)
    parser.add_argument("--static-match-distance-m", type=float, default=2.5)
    parser.add_argument("--cruise-speed-mps", type=float, default=8.0)
    parser.add_argument("--forward-horizon-m", type=float, default=60.0)
    parser.add_argument("--waypoint-step-m", type=float, default=5.0)
    parser.add_argument("--route-option", default=None)
    parser.add_argument("--preferred-lane", default=None)
    parser.add_argument("--route-priority", default="normal")
    parser.add_argument("--route-manifest", default=None)
    parser.add_argument("--reset-state-on-stage-failure", action="store_true")

    parser.add_argument("--sampling-radius-m", type=float, default=2.0)
    parser.add_argument("--prepare-offset-m", type=float, default=0.75)
    parser.add_argument("--lane-change-success-hold-ticks", type=int, default=4)
    parser.add_argument("--max-lane-change-ticks", type=int, default=45)
    parser.add_argument("--lane-change-min-lateral-shift-m", type=float, default=0.50)
    parser.add_argument("--lane-change-min-lateral-shift-fraction", type=float, default=0.15)
    parser.add_argument("--stop-hold-ticks", type=int, default=10)
    parser.add_argument("--stop-full-stop-speed-threshold-mps", type=float, default=0.35)
    parser.add_argument("--stop-near-distance-m", type=float, default=8.0)
    parser.add_argument("--stop-hard-brake-distance-m", type=float, default=3.0)
    parser.add_argument("--soft-stale-ticks", type=int, default=8)
    parser.add_argument("--hard-stale-ticks", type=int, default=24)

    parser.add_argument("--record-mp4", action="store_true")
    parser.add_argument("--recording-path", default=None)
    parser.add_argument("--recording-camera-mode", choices=("chase", "hood", "topdown"), default="chase")
    parser.add_argument("--recording-width", type=int, default=1280)
    parser.add_argument("--recording-height", type=int, default=720)
    parser.add_argument("--recording-fov", type=float, default=110.0)
    parser.add_argument("--recording-fps", type=float, default=0.0)
    parser.add_argument("--recording-overlay", dest="recording_overlay", action="store_true")
    parser.add_argument("--recording-no-overlay", dest="recording_overlay", action="store_false")
    parser.set_defaults(recording_overlay=True)

    parser.add_argument("--stop-alignment-enabled", action="store_true")
    parser.add_argument("--stop-alignment-sample-name", default=None)
    parser.add_argument("--stop-alignment-request-frame", type=int, default=0)
    parser.add_argument("--stop-alignment-expected-gap-m", type=float, default=0.0)
    parser.add_argument("--stop-alignment-gap-tolerance-m", type=float, default=3.0)
    parser.add_argument("--stop-alignment-source", default="manifest_distance_minus_expected_stop_gap")

    parser.add_argument("--fallback-probe-tick", type=int, default=-1)
    parser.add_argument("--fallback-probe-trigger", choices=("hard_timeout", "soft_timeout"), default=None)
    parser.add_argument(
        "--fallback-perturbation-mode",
        choices=("disabled", "route_uncertainty"),
        default="disabled",
    )
    parser.add_argument("--fallback-perturbation-start-tick", type=int, default=-1)
    parser.add_argument("--shadow-mode", action="store_true")
    parser.add_argument(
        "--control-authority-sandbox-mode",
        choices=("disabled", "no_motion_handoff", "bounded_motion_handoff", "bounded_motion_shadow_authority", "full_shadow_authority"),
        default="disabled",
    )
    parser.add_argument("--control-authority-sandbox-handoff-tick", type=int, default=-1)
    parser.add_argument("--control-authority-sandbox-handoff-duration-ticks", type=int, default=-1)
    parser.add_argument("--control-authority-sandbox-max-speed-mps", type=float, default=0.35)
    parser.add_argument("--control-authority-sandbox-min-speed-mps", type=float, default=0.05)

    args = parser.parse_args()
    if _mp4_recording_disabled() and bool(getattr(args, "record_mp4", False)):
        print("[Stage4] MP4 recording disabled by AGENTAI_DISABLE_MP4; ignoring --record-mp4")
        args.record_mp4 = False
    if args.stage1_num_samples <= 0:
        args.stage1_num_samples = int(args.warmup_ticks) + int(args.num_ticks) + 16
    return args


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging()
    ensure_carla_pythonapi(args.carla_pythonapi_root)
    from stage4.online_orchestrator import Stage4OnlineOrchestrator

    orchestrator = Stage4OnlineOrchestrator(args)
    orchestrator.run()


if __name__ == "__main__":
    main()
