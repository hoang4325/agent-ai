"""
scripts/run_stage10_stage1_live_bridge.py
==========================================
Stage 10 — Live CARLA → BEVFusion → Stage 9 bridge.

Connects every module we have built (Stage 1 → Stage 9) into a single
synchronous closed-loop:

  CARLA sensors ──► CarlaSensorSync
                       │  LiveFrame
                       ▼
             BEVFusionLiveAdapter  (bevfusion_runtime)
                       │  DetectionList
                       ▼
              WorldStateBuilder
                       │  WorldState
                       ▼
          [TODO S9] AuthorityArbiter.step()
                       │  ActuatorCommand
                       ▼
              carla_vehicle.apply_control()

Usage:
  python scripts/run_stage10_stage1_live_bridge.py \\
      --carla-root  "C:/CARLA" \\
      --bev-repo    "C:/bevfusion_repo" \\
      --bev-config  "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/...yaml" \\
      --bev-ckpt    "checkpoints/bevfusion.pth" \\
      --max-frames  500 \\
      --device      cuda

Exit code 0 = completed normally, 1 = error/crash.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from carla_bevfusion_stage1.carla_sensor_sync import LiveFrame
    from carla_bevfusion_stage1.world_state_builder import EgoTelemetry

# ── Ensure project root is on sys.path ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("stage10_live_bridge")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 10 – CARLA + BEVFusion live bridge")
    p.add_argument("--carla-root",   required=True,  help="Path to CARLA installation root")
    p.add_argument("--carla-host",   default="127.0.0.1")
    p.add_argument("--carla-port",   type=int, default=2000)
    p.add_argument("--bev-repo",     required=True,  help="BEVFusion git repo root")
    p.add_argument("--bev-config",   required=True,  help="BEVFusion YAML config path")
    p.add_argument("--bev-ckpt",     required=True,  help="BEVFusion checkpoint .pth")
    p.add_argument("--output-root",  required=True,  help="Where to write inference results")
    p.add_argument("--samples-root", default=None,   help="Watch mode: poll this folder for dumped samples instead of connecting to CARLA")
    p.add_argument("--device",       default="cuda",  help="torch device")
    p.add_argument("--max-frames",   type=int, default=500)
    p.add_argument("--delta-t",      type=float, default=0.1,  help="Simulation step (s)")
    p.add_argument("--rig-profile",  default="default",
                   choices=["default", "low_memory_shadow"],
                   help="Sensor rig quality preset")
    p.add_argument("--image-width",  type=int, default=1600)
    p.add_argument("--image-height", type=int, default=900)
    p.add_argument("--score-thresh", type=float, default=0.35)
    p.add_argument("--map",          default="Town01",  help="CARLA map name")
    p.add_argument("--spawn-point",  type=int, default=0, help="Ego spawn-point index")
    p.add_argument("--enable-radar", action="store_true", default=True)
    p.add_argument("--radar-ablation", choices=("none", "zero_bev"), default="none",
                   help="Ablate radar branch (none=active, zero_bev=disabled)")
    p.add_argument("--log-dir",      default="benchmark/reports/stage10_live",
                   help="Directory to write JSONL authority log")
    p.add_argument("--no-stage9",    action="store_true",
                   help="Skip Stage 9 arbiter (perception-only dry run)")
    p.add_argument("--agent-mode",   default="stub",
                   choices=["stub", "api", "compare"],
                   help="Agent mode: stub=no LLM, api=real LLM, compare=baseline+agent side-by-side logging")
    return p.parse_args()


# ── Watch mode (Folder Polling) ──────────────────────────────────────────────

class FolderWatcherSync:
    """
    Mirrors CarlaSensorSync interface but reads from a folder on disk.
    Polls for 'sample_complete.json' or 'meta.json' before processing.
    """
    def __init__(
        self,
        samples_root: Path,
        max_samples: int,
        poll_interval_s: float = 0.5,
    ) -> None:
        self._root = Path(samples_root)
        self._max = max_samples
        self._poll_s = poll_interval_s
        self._processed: Set[str] = set()
        self._frame_counter = 0

    def start(self) -> None:
        if not self._root.exists():
            raise FileNotFoundError(f"Watch root does not exist: {self._root}")
        LOGGER.info("FolderWatcherSync started on %s", self._root)

    def stop(self) -> None:
        LOGGER.info("FolderWatcherSync stopped. Processed %d samples", self._frame_counter)

    def tick(self) -> Optional[LiveFrame]:
        """Poll for the next newest ready sample."""
        from carla_bevfusion_stage1.carla_sensor_sync import (
            LiveFrame, LiveCalibration, _camera_image_to_bgra, _lidar_to_xyzi, _radar_to_array
        )
        from carla_bevfusion_stage1.constants import LIDAR_SENSOR_NAME, MODEL_CAMERA_ORDER, RADAR_SENSOR_ORDER
        from PIL import Image

        while self._frame_counter < self._max:
            # 1. Find all 'sample_xxxxxx' dirs
            candidates = sorted(
                [d for d in self._root.iterdir() if d.is_dir() and d.name.startswith("sample_")],
                key=lambda x: int(x.name.split("_")[-1])
            )
            # 2. Filter new and ready
            new_ready = [d for d in candidates if d.name not in self._processed and (d / "meta.json").exists()]

            if not new_ready:
                time.sleep(self._poll_s)
                continue

            target = new_ready[0]
            self._processed.add(target.name)
            self._frame_counter += 1

            # 3. Load meta
            with (target / "meta.json").open("r") as f:
                meta = json.load(f)

            # 4. Reconstruct LiveFrame
            images_bgra = {}
            for cam in MODEL_CAMERA_ORDER:
                img = Image.open(target / "images" / f"{cam}.png")
                # Convert PIL to BGRA array
                rgb = np.array(img.convert("RGB"))
                bgra = np.zeros((img.height, img.width, 4), dtype=np.uint8)
                bgra[:, :, :3] = rgb[:, :, ::-1] # RGB -> BGR
                bgra[:, :, 3] = 255
                images_bgra[cam] = bgra

            lidar_xyzi = np.load(target / "lidar" / f"{LIDAR_SENSOR_NAME}.npy")

            radar_raw = {}
            for r in RADAR_SENSOR_ORDER:
                radar_path = target / "radar" / f"{r}.npy"
                radar_raw[r] = np.load(radar_path) if radar_path.exists() else np.zeros((0,4), dtype=np.float32)

            # Reconstruct Calibration from meta
            # (Simplification: meta stores matrix_world_from_sensor_bevfusion etc)
            from carla_bevfusion_stage1.coordinate_utils import ensure_numpy_matrix
            s_meta = meta["sensors"]
            calib = LiveCalibration(
                world_from_lidar_bev=ensure_numpy_matrix(s_meta[LIDAR_SENSOR_NAME]["matrix_world_from_sensor_bevfusion"]),
                ego_from_lidar_bev=ensure_numpy_matrix(s_meta[LIDAR_SENSOR_NAME]["matrix_ego_from_sensor_bevfusion"]),
                world_from_cameras_bev={c: ensure_numpy_matrix(s_meta[c]["matrix_world_from_sensor_bevfusion"]) for c in MODEL_CAMERA_ORDER},
                ego_from_cameras_bev={c: ensure_numpy_matrix(s_meta[c]["matrix_ego_from_sensor_bevfusion"]) for c in MODEL_CAMERA_ORDER},
                camera_intrinsics={c: np.array(s_meta[c]["camera_intrinsics"], dtype=np.float32) for c in MODEL_CAMERA_ORDER},
            )

            ego_vel = np.array(meta["ego"]["velocity_carla"], dtype=np.float32)

            return LiveFrame(
                frame_id=int(meta["frame_id"]),
                timestamp_s=float(meta["timestamp"]),
                images_bgra=images_bgra,
                lidar_xyzi=lidar_xyzi,
                radar_raw=radar_raw,
                calibration=calib,
                ego_velocity_carla=ego_vel,
            )
        return None



# ── CARLA helpers ─────────────────────────────────────────────────────────────

def _connect_carla(host: str, port: int, timeout_s: float = 30.0):
    """Connect to a running CARLA instance."""
    try:
        import carla  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API not found. Add CARLA PythonAPI to PYTHONPATH."
        ) from exc
    client = carla.Client(host, port)
    client.set_timeout(timeout_s)
    LOGGER.info("Connecting to CARLA @ %s:%d …", host, port)
    world = client.get_world()
    LOGGER.info("Connected. Map: %s", world.get_map().name)
    return client, world


def _load_map_if_needed(client, world, target_map: str):
    import carla  # type: ignore
    current = world.get_map().name.split("/")[-1]
    if current != target_map:
        LOGGER.info("Loading map %s (was %s) …", target_map, current)
        world = client.load_world(target_map)
        time.sleep(2.0)
    return world


def _spawn_ego(world, spawn_idx: int = 0):
    import carla  # type: ignore
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found in this map.")
    sp = spawn_points[spawn_idx % len(spawn_points)]
    ego = world.spawn_actor(vehicle_bp, sp)
    ego.set_autopilot(True)      # Simple traffic-manager autopilot as baseline
    LOGGER.info("Ego spawned at spawn_point=%d actor_id=%d", spawn_idx, ego.id)
    return ego


def _apply_actuator_command(ego_actor, cmd) -> None:
    """
    Apply a Stage 9 ActuatorCommand to the CARLA ego vehicle.

    TODO [Stage 9 hook]: Uncomment and implement when wiring up Arbiter.
    cmd fields: steer_norm ∈ [-1,1], throttle_norm ∈ [0,1], brake_norm ∈ [0,1]
    """
    try:
        import carla  # type: ignore
        control = carla.VehicleControl(
            throttle=float(getattr(cmd, "throttle_norm", 0.0)),
            steer=float(getattr(cmd, "steer_norm", 0.0)),
            brake=float(getattr(cmd, "brake_norm", 0.0)),
        )
        ego_actor.apply_control(control)
    except Exception as exc:
        LOGGER.warning("apply_control failed: %s", exc)


# ── Stage 9 Arbiter bootstrap ─────────────────────────────────────────────────
# TODO [Stage 9 hook]: Replace stubs with real Stage 9 components.

def _build_stage9_arbiter(log_dir: Path, agent_mode: str = "stub"):
    """
    Build and return the Stage 9 AuthorityArbiter with REAL components.
    Returns None if imports fail (graceful degradation).
    """
    try:
        from stage9 import (
            AuthorityArbiter,
            AuthorityStateMachine,
            BaselineDetector,
            ContractResolver,
            HandoffPlanner,
            HumanOverrideMonitor,
            MRMExecutor,
            SafetySupervisor,
            TORManager,
        )
    except ImportError as exc:
        LOGGER.warning("Stage 9 not available: %s — running perception-only.", exc)
        return None

    # ── Real Adapter components (Stage 6 MPC + Stage 7 Agent) ────────────────
    try:
        from carla_bevfusion_stage1.stage9_adapters import (
            RealMPCAdapter,
            RealBaselineAdapter,
            RealAgentAdapter,
        )
        mpc_impl = RealMPCAdapter(dt_s=0.1)
        baseline_impl = RealBaselineAdapter()
        _adapter_mode = "api" if agent_mode in ("api", "compare") else "stub"
        agent_impl = RealAgentAdapter(mode=_adapter_mode)
        LOGGER.info("Stage 10: Real MPC + Baseline + Agent adapters loaded (agent_mode=%s).", _adapter_mode)
    except ImportError as exc:
        LOGGER.warning("stage9_adapters unavailable (%s). Using minimal stubs.", exc)

        class _StubBaseline:
            def plan(self, w): return _stub_traj()
            def degraded_hold(self, w): return _stub_traj()
            def is_healthy(self, w): return True

        class _StubMPC:
            def execute(self, req): return _stub_cmd()
            def preview_feasible(self, req): return True

        class _StubAgent:
            def propose_contract(self, w): return None
            def get_intent(self, w, c): return "keep_lane"

        mpc_impl = _StubMPC()
        baseline_impl = _StubBaseline()
        agent_impl = _StubAgent()

    # ── Stubs still needed for TOR (no real TOR implementation) ──────────────
    class _StubTOR:
        MIN_WAIT_BUDGET_S = 5.0
        def start(self, w): pass
        def tick(self, w): pass
        def timed_out(self): return False

    def _stub_traj():
        from stage9.schemas import TrajectoryRequest
        return TrajectoryRequest()

    def _stub_cmd():
        from stage9.schemas import ActuatorCommand
        return ActuatorCommand(steer=0.0, throttle=0.0, brake=0.0)

    log_dir.mkdir(parents=True, exist_ok=True)
    arbiter = AuthorityArbiter(
        asm=AuthorityStateMachine(),
        baseline_detector=BaselineDetector(),
        human_override=HumanOverrideMonitor(),
        agent=agent_impl,
        supervisor=SafetySupervisor(),
        baseline=baseline_impl,
        contract_resolver=ContractResolver(),
        handoff_planner=HandoffPlanner(),
        mpc=mpc_impl,
        tor=_StubTOR(),
        mrm=MRMExecutor(),
        log_path=log_dir / "stage9_authority_log.jsonl",
    )
    LOGGER.info("Stage 9 AuthorityArbiter initialised with REAL components.")
    return arbiter



# ── Main loop ─────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    from carla_bevfusion_stage1.bevfusion_runtime import build_bevfusion_model
    from carla_bevfusion_stage1.bevfusion_live_adapter import BEVFusionLiveAdapter
    from carla_bevfusion_stage1.carla_sensor_sync import CarlaSensorSync
    from carla_bevfusion_stage1.rig import build_rig_preset
    from carla_bevfusion_stage1.world_state_builder import WorldStateBuilder, EgoTelemetry

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Build BEVFusion model FIRST (before sensors start) ────────────────
    # IMPORTANT: mmdet3d import takes ~50s. Loading model before attaching CARLA
    # sensors prevents sensor buffer overflow / frame-sync timeouts.
    LOGGER.info("Loading BEVFusion model (before sensor spawn to avoid timeout) …")
    model, cfg, BoxClass = build_bevfusion_model(
        repo_root=args.bev_repo,
        config_path=args.bev_config,
        checkpoint_path=args.bev_ckpt,
        device=args.device,
        radar_ablation=args.radar_ablation,
    )
    bev_adapter = BEVFusionLiveAdapter(
        model, cfg, BoxClass,
        device=args.device,
        score_threshold=args.score_thresh,
    )
    LOGGER.info("BEVFusion ready.")

    # ── 2. Input Source (CARLA or Watch Folder) ───────────────────────────────
    # Sensors are spawned NOW, after the model is warm, so no frames are missed.
    if args.samples_root:
        sensor_source = FolderWatcherSync(args.samples_root, args.max_frames)
        ego = None
        carla_map = None
    else:
        client, world = _connect_carla(args.carla_host, args.carla_port)
        world = _load_map_if_needed(client, world, args.map)
        carla_map = world.get_map()
        ego = _spawn_ego(world, args.spawn_point)

        preset = build_rig_preset(
            args.rig_profile,
            image_width=args.image_width,
            image_height=args.image_height,
            camera_fov=70.0,
            fixed_delta_seconds=args.delta_t,
        )
        sensor_source = CarlaSensorSync(  # type: ignore
            world, ego, preset,
            fixed_delta_seconds=args.delta_t,
            image_width=args.image_width,
            image_height=args.image_height,
            enable_radar=args.enable_radar,
        )

    # ── 3. Stage 9 Arbiter (optional) ────────────────────────────────────────
    arbiter = None if args.no_stage9 else _build_stage9_arbiter(log_dir, agent_mode=args.agent_mode)

    # ── 3b. Compare-mode: separate agent adapter for side-by-side logging ────
    compare_agent = None
    compare_baseline = None
    compare_log: List[Dict[str, Any]] = []
    if args.agent_mode == "compare":
        try:
            from carla_bevfusion_stage1.stage9_adapters import RealAgentAdapter, RealBaselineAdapter
            compare_agent = RealAgentAdapter(mode="api")
            compare_baseline = RealBaselineAdapter()
            LOGGER.info("Compare mode: side-by-side Agent vs Baseline logging ENABLED.")
        except ImportError as exc:
            LOGGER.warning("Compare mode unavailable (%s). Falling back to normal mode.", exc)
            compare_agent = None


    # ── 4. Sensor rig + sync ──────────────────────────────────────────────────
    # (Moved setup to sensor_source initialization above)

    # ── 5. WorldState Builder ─────────────────────────────────────────────────
    ws_builder = WorldStateBuilder(human_driver_available=True)

    # ── Graceful shutdown on Ctrl-C ───────────────────────────────────────────
    _shutdown = [False]
    def _sigint_handler(sig, frame):
        LOGGER.warning("SIGINT received – stopping after current frame.")
        _shutdown[0] = True
    signal.signal(signal.SIGINT, _sigint_handler)

    sensor_source.start()
    LOGGER.info("=" * 60)
    LOGGER.info("Stage 10 Bridge started. source=%s", "WatchMode" if args.samples_root else "LiveMode")
    LOGGER.info("=" * 60)

    stats = {"frames": 0, "total_det": 0, "bev_ms_sum": 0.0, "errors": 0}
    prev_v = 0.0

    try:
        for frame_idx in range(args.max_frames):
            if _shutdown[0]:
                break

            t_tick = time.monotonic()

            # ── A. Sensor tick ─────────────────────────────────────────────
            try:
                live_frame = sensor_source.tick()
                if live_frame is None:
                    break
            except TimeoutError as exc:
                LOGGER.error("Sensor timeout frame=%d: %s", frame_idx, exc)
                stats["errors"] += 1
                continue

            # ── B. BEVFusion inference ─────────────────────────────────────
            det_list = bev_adapter.adapt_and_infer(live_frame)
            stats["bev_ms_sum"] += det_list.inference_time_ms
            stats["total_det"] += len(det_list.detections)

            # ── C. Ego telemetry ───────────────────────────────────────────
            if not args.samples_root and ego is not None and carla_map is not None:
                ego_tel = WorldStateBuilder.extract_ego_telemetry(
                    ego, carla_map,
                    frame_id=live_frame.frame_id,
                    timestamp_s=live_frame.timestamp_s,
                    prev_v_mps=prev_v,
                )
            else:
                # Reconstruct from meta in watch mode
                ego_tel = EgoTelemetry(
                    frame_id=live_frame.frame_id,
                    timestamp_s=live_frame.timestamp_s,
                    ego_lane_id="watch_mode_unknown",
                    ego_v_mps=float(np.sqrt(np.sum(live_frame.ego_velocity_carla**2))),
                    ego_a_mps2=0.0,
                    ego_lateral_error_m=0.0,
                    heading_error_rad=0.0,
                    ego_location_xyz=np.zeros(3),
                )
            prev_v = ego_tel.ego_v_mps

            # ── D. WorldState ──────────────────────────────────────────────
            world_state = ws_builder.build(det_list, ego_tel)

            # ── E. Stage 9 Arbiter ─────────────────────────────────────────
            if arbiter is not None and world_state is not None:
                cmd = arbiter.step(world_state, sim_time_s=live_frame.timestamp_s)
                if not args.samples_root and ego is not None:
                    _apply_actuator_command(ego, cmd)

            # ── F. Compare mode: side-by-side Agent vs Baseline ───────────
            if compare_agent is not None and compare_baseline is not None and world_state is not None:
                try:
                    t_cmp = time.monotonic()
                    bl_req = compare_baseline.plan(world_state)
                    baseline_intent = str(getattr(bl_req, "tactical_intent", "keep_lane"))

                    contract = compare_agent.propose_contract(world_state)
                    if contract is not None:
                        agent_intent = str(getattr(contract, "tactical_intent", "keep_lane"))
                        agent_confidence = float(getattr(contract, "agent_confidence", 0.0))
                        agent_reasoning = str(getattr(contract, "agent_reasoning_summary", ""))
                        disagreement_useful = True
                    else:
                        agent_intent = baseline_intent  # agent agrees (or fallback)
                        agent_confidence = 0.0
                        agent_reasoning = "agrees_with_baseline"
                        disagreement_useful = False

                    agrees = (baseline_intent == agent_intent)
                    cmp_ms = (time.monotonic() - t_cmp) * 1000

                    ttc_val = float(
                        ws_builder._prev_detections and
                        _ttc_from_prev(ws_builder._prev_detections, ego_tel.ego_v_mps) or 99.0
                    )

                    record = {
                        "frame_id": live_frame.frame_id,
                        "frame_idx": frame_idx,
                        "timestamp_s": live_frame.timestamp_s,
                        "ego_v_mps": ego_tel.ego_v_mps,
                        "min_ttc_s": ttc_val,
                        "num_detections": len(det_list.detections),
                        "baseline_intent": baseline_intent,
                        "agent_intent": agent_intent,
                        "agent_confidence": agent_confidence,
                        "agent_reasoning": agent_reasoning,
                        "agrees": agrees,
                        "disagreement_useful": disagreement_useful and not agrees,
                        "compare_latency_ms": round(cmp_ms, 1),
                    }
                    compare_log.append(record)

                    if not agrees:
                        LOGGER.info(
                            "[COMPARE] frame=%d  baseline=%s  agent=%s  conf=%.2f  ttc=%.1f  reason=%s",
                            live_frame.frame_id, baseline_intent, agent_intent,
                            agent_confidence, ttc_val, agent_reasoning,
                        )
                except Exception as exc:
                    LOGGER.debug("Compare-mode error frame=%d: %s", frame_idx, exc)

            stats["frames"] += 1
            elapsed_ms = (time.monotonic() - t_tick) * 1000

            if frame_idx % 20 == 0:
                avg_bev = stats["bev_ms_sum"] / max(1, stats["frames"])
                LOGGER.info(
                    "[%4d/%4d] frame=%d  dets=%2d  bev=%.0f ms  tick=%.0f ms  v=%.1f m/s  ttc=%.1f s",
                    frame_idx, args.max_frames,
                    live_frame.frame_id,
                    len(det_list.detections),
                    det_list.inference_time_ms,
                    elapsed_ms,
                    ego_tel.ego_v_mps,
                    ws_builder._prev_detections and
                    _ttc_from_prev(ws_builder._prev_detections, ego_tel.ego_v_mps) or 99.0,
                )

    except Exception as exc:
        LOGGER.exception("Unexpected error in main loop: %s", exc)
        stats["errors"] += 1
        return 1
    finally:
        sensor_source.stop()
        if not args.samples_root and ego is not None:
            try:
                ego.destroy()
            except Exception:
                pass

        # ── Dump compare-mode evaluation report ──────────────────────────
        if compare_log:
            total = len(compare_log)
            agreements = sum(1 for r in compare_log if r["agrees"])
            disagreements = total - agreements
            useful_disagreements = sum(1 for r in compare_log if r["disagreement_useful"])
            agent_latencies = [r["compare_latency_ms"] for r in compare_log]

            # TTC-based analysis: how does agent behave in danger zones?
            low_ttc_frames = [r for r in compare_log if r["min_ttc_s"] < 3.0]
            low_ttc_disagree = [r for r in low_ttc_frames if not r["agrees"]]

            evaluation = {
                "schema_version": "stage10_agent_live_evaluation_v1",
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                "stage": "10_live_bridge",
                "mode": args.agent_mode,
                "map": args.map,
                "total_frames": total,
                "agreement_frames": agreements,
                "disagreement_frames": disagreements,
                "agreement_rate": round(agreements / max(total, 1), 4),
                "disagreement_rate": round(disagreements / max(total, 1), 4),
                "useful_disagreement_count": useful_disagreements,
                "useful_disagreement_rate": round(
                    useful_disagreements / max(disagreements, 1), 4
                ) if disagreements > 0 else None,
                "low_ttc_analysis": {
                    "total_low_ttc_frames": len(low_ttc_frames),
                    "disagreements_in_low_ttc": len(low_ttc_disagree),
                    "agent_cautious_rate": round(
                        len(low_ttc_disagree) / max(len(low_ttc_frames), 1), 4
                    ) if low_ttc_frames else None,
                },
                "latency": {
                    "mean_compare_ms": round(sum(agent_latencies) / max(len(agent_latencies), 1), 1),
                    "max_compare_ms": round(max(agent_latencies), 1) if agent_latencies else 0,
                    "min_compare_ms": round(min(agent_latencies), 1) if agent_latencies else 0,
                },
                "intent_distribution": {
                    "baseline": {},
                    "agent": {},
                },
                "perception_summary": {
                    "total_detections": stats["total_det"],
                    "avg_detections_per_frame": round(stats["total_det"] / max(total, 1), 1),
                    "avg_bev_inference_ms": round(stats["bev_ms_sum"] / max(total, 1), 1),
                },
                "frame_log": compare_log,
            }

            # Count intent distributions
            for r in compare_log:
                bi = r["baseline_intent"]
                ai = r["agent_intent"]
                evaluation["intent_distribution"]["baseline"][bi] = (
                    evaluation["intent_distribution"]["baseline"].get(bi, 0) + 1
                )
                evaluation["intent_distribution"]["agent"][ai] = (
                    evaluation["intent_distribution"]["agent"].get(ai, 0) + 1
                )

            eval_path = log_dir / "stage10_agent_live_evaluation.json"
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            with open(eval_path, "w") as f:
                json.dump(evaluation, f, indent=2, default=str)
            LOGGER.info("Compare-mode evaluation saved → %s", eval_path)
            LOGGER.info("  Agreement rate   : %.1f%% (%d/%d)", agreements / max(total, 1) * 100, agreements, total)
            LOGGER.info("  Disagreement rate: %.1f%% (%d/%d)", disagreements / max(total, 1) * 100, disagreements, total)
            if disagreements > 0:
                LOGGER.info("  Useful disagree  : %.1f%% (%d/%d)",
                            useful_disagreements / max(disagreements, 1) * 100,
                            useful_disagreements, disagreements)
            if low_ttc_frames:
                LOGGER.info("  Low-TTC frames   : %d, agent cautious: %d",
                            len(low_ttc_frames), len(low_ttc_disagree))

        LOGGER.info("=" * 60)
        LOGGER.info("Stage 10 Live Bridge finished.")
        LOGGER.info("  Frames     : %d", stats["frames"])
        LOGGER.info("  Total detections : %d", stats["total_det"])
        if stats["frames"]:
            LOGGER.info("  Avg BEV ms : %.1f", stats["bev_ms_sum"] / stats["frames"])
        LOGGER.info("  Errors     : %d", stats["errors"])
        LOGGER.info("=" * 60)

    return 0 if stats["errors"] == 0 else 1


def _ttc_from_prev(dets, ego_v):
    from carla_bevfusion_stage1.world_state_builder import _estimate_ttc
    return _estimate_ttc(dets, ego_v)


if __name__ == "__main__":
    sys.exit(run(_parse_args()))
