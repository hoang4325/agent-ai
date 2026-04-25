from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from stage3c.execution_contract import ExecutionRequest
from stage3c.execution_monitor import ExecutionMonitor
from stage3c.local_planner_bridge import LocalPlannerBridge, PlannerStepResult
from stage3c.video_recorder import Stage3CVideoRecorder

from .arbitration import ArbitrationDecision, arbitrate_request

LOGGER = logging.getLogger("stage4.execution_runtime")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


@dataclass
class _PendingExecutionTick:
    tick_id: int
    arbitration: ArbitrationDecision
    planner_step: PlannerStepResult
    latest_behavior_frame_id: int | None
    started_lane_change: bool
    prepare_latency_ms: float


class ExecutionRuntime:
    def __init__(
        self,
        *,
        vehicle: Any,
        map_inst: Any,
        pythonapi_root: str | Path,
        output_dir: str | Path,
        fixed_delta_seconds: float = 0.10,
        sampling_radius_m: float = 2.0,
        prepare_offset_m: float = 0.75,
        lane_change_success_hold_ticks: int = 4,
        max_lane_change_ticks: int = 45,
        lane_change_min_lateral_shift_m: float = 0.50,
        lane_change_min_lateral_shift_fraction: float = 0.15,
        stop_hold_ticks: int = 10,
        stop_full_stop_speed_threshold_mps: float = 0.35,
        stop_near_distance_m: float = 8.0,
        stop_hard_brake_distance_m: float = 3.0,
        recording_enabled: bool = False,
        recording_path: str | Path | None = None,
        recording_fps: float = 10.0,
        recording_width: int = 1280,
        recording_height: int = 720,
        recording_fov: float = 110.0,
        recording_camera_mode: str = "chase",
        recording_overlay: bool = True,
        soft_stale_ticks: int = 8,
        hard_stale_ticks: int = 24,
    ) -> None:
        self.vehicle = vehicle
        self.map_inst = map_inst
        self.output_dir = Path(output_dir)
        self.timeline_path = self.output_dir / "execution_timeline.jsonl"
        self.lane_change_events_path = self.output_dir / "lane_change_events.jsonl"
        self.timeline_path.unlink(missing_ok=True)
        self.lane_change_events_path.unlink(missing_ok=True)

        self.bridge = LocalPlannerBridge(
            vehicle=vehicle,
            map_inst=map_inst,
            pythonapi_root=pythonapi_root,
            fixed_delta_seconds=fixed_delta_seconds,
            sampling_radius_m=sampling_radius_m,
            prepare_offset_m=prepare_offset_m,
            lane_change_success_hold_ticks=lane_change_success_hold_ticks,
            max_lane_change_ticks=max_lane_change_ticks,
            lane_change_min_lateral_shift_m=lane_change_min_lateral_shift_m,
            lane_change_min_lateral_shift_fraction=lane_change_min_lateral_shift_fraction,
            stop_hold_ticks=stop_hold_ticks,
            stop_full_stop_speed_threshold_mps=stop_full_stop_speed_threshold_mps,
            stop_near_distance_m=stop_near_distance_m,
            stop_hard_brake_distance_m=stop_hard_brake_distance_m,
        )
        self.monitor = ExecutionMonitor()
        self.last_effective_request: ExecutionRequest | None = None
        self.latest_execution_state: Dict[str, Any] | None = None
        self.soft_stale_ticks = int(soft_stale_ticks)
        self.hard_stale_ticks = int(hard_stale_ticks)
        self._pending_tick: _PendingExecutionTick | None = None

        self.video_recorder: Stage3CVideoRecorder | None = None
        if recording_enabled:
            self.video_recorder = Stage3CVideoRecorder(
                world=vehicle.get_world(),
                vehicle=vehicle,
                output_path=recording_path or (self.output_dir / "video" / "closed_loop.mp4"),
                fps=recording_fps,
                width=recording_width,
                height=recording_height,
                fov=recording_fov,
                mode=recording_camera_mode,
                overlay=recording_overlay,
            )

    def prepare_tick(
        self,
        *,
        tick_id: int,
        snapshot_before: Any,
        proposed_request: ExecutionRequest | None,
        perception_stale_ticks: int,
        last_behavior_frame_id: int | None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        arbitration = arbitrate_request(
            proposed_request=proposed_request,
            last_effective_request=self.last_effective_request,
            active_lane_change=self.bridge.active_lane_change,
            perception_stale_ticks=int(perception_stale_ticks),
            soft_stale_ticks=self.soft_stale_ticks,
            hard_stale_ticks=self.hard_stale_ticks,
            last_behavior_frame=last_behavior_frame_id,
        )
        had_active_lane_change = self.bridge.active_lane_change is not None
        planner_step = self.bridge.step(arbitration.effective_request, snapshot_before)
        self.vehicle.apply_control(planner_step.control)
        self._pending_tick = _PendingExecutionTick(
            tick_id=int(tick_id),
            arbitration=arbitration,
            planner_step=planner_step,
            latest_behavior_frame_id=last_behavior_frame_id,
            started_lane_change=(not had_active_lane_change) and (self.bridge.active_lane_change is not None),
            prepare_latency_ms=(time.perf_counter() - start) * 1000.0,
        )
        return {
            "arbitration_action": arbitration.action,
            "arbitration_notes": list(arbitration.notes),
            "effective_request": arbitration.effective_request.to_dict(),
        }

    def _build_overlay_lines(self, timeline_record: Dict[str, Any]) -> List[str]:
        execution_request = timeline_record["execution_request"]
        execution_state = timeline_record["execution_state"]
        junction_tag = "junction" if execution_state.get("is_in_junction") else "road"
        return [
            (
                f"tick={timeline_record.get('tick_id')} sample={timeline_record['sample_name']} "
                f"req_frame={timeline_record['request_frame']} carla_frame={timeline_record['carla_frame']}"
            ),
            (
                f"requested={execution_request['requested_behavior']} "
                f"executed={execution_state['current_behavior']} status={execution_state['execution_status']}"
            ),
            (
                f"lane={execution_state['current_lane_id']} -> {execution_state['target_lane_id']} "
                f"progress={execution_state['lane_change_progress']:.2f} mode={junction_tag}"
            ),
            (
                f"speed={execution_state['speed_mps']:.2f} mps "
                f"target={execution_request['target_speed_mps']:.2f} mps "
                f"route={execution_request.get('route_option', 'unknown')}"
            ),
        ]

    def finalize_tick(self, *, snapshot_after: Any) -> Dict[str, Any]:
        if self._pending_tick is None:
            raise RuntimeError("ExecutionRuntime.finalize_tick() called before prepare_tick().")

        pending = self._pending_tick
        finalize_start = time.perf_counter()

        if pending.started_lane_change and self.bridge.active_lane_change is not None:
            event = self.monitor.record_lane_change_start(
                snapshot=snapshot_after,
                tracker=self.bridge.active_lane_change,
            )
            _append_jsonl(self.lane_change_events_path, event)

        completed_tracker = self.bridge.post_tick_update(snapshot_after)
        tracker_for_state = completed_tracker or self.bridge.active_lane_change
        timeline_record = self.monitor.record(
            execution_request=pending.arbitration.effective_request,
            planner_step=pending.planner_step,
            snapshot=snapshot_after,
            vehicle=self.vehicle,
            map_inst=self.map_inst,
            lane_change_tracker=tracker_for_state,
        )
        execution_finalize_latency_ms = (time.perf_counter() - finalize_start) * 1000.0
        planner_diagnostics = dict(timeline_record.get("planner_diagnostics") or {})
        planner_diagnostics.update(
            {
                "planner_prepare_latency_ms": float(pending.prepare_latency_ms),
                "execution_finalize_latency_ms": float(execution_finalize_latency_ms),
                "planner_execution_latency_ms": float(pending.prepare_latency_ms + execution_finalize_latency_ms),
            }
        )
        timeline_record["planner_diagnostics"] = planner_diagnostics
        timeline_record["tick_id"] = int(pending.tick_id)
        timeline_record["arbitration"] = pending.arbitration.to_dict()
        timeline_record["behavior_frame_id"] = pending.latest_behavior_frame_id
        _append_jsonl(self.timeline_path, timeline_record)

        if completed_tracker is not None:
            event = self.monitor.record_lane_change_terminal(
                snapshot=snapshot_after,
                tracker=completed_tracker,
                current_lane_id=timeline_record["execution_state"]["current_lane_id"],
            )
            _append_jsonl(self.lane_change_events_path, event)

        if self.video_recorder is not None:
            wrote = self.video_recorder.write_frame(
                carla_frame=int(timeline_record["carla_frame"]),
                overlay_lines=self._build_overlay_lines(timeline_record),
            )
            if not wrote:
                LOGGER.warning(
                    "Stage4 video recorder missed carla_frame=%d tick=%d",
                    int(timeline_record["carla_frame"]),
                    int(pending.tick_id),
                )

        total_execution_ms = pending.prepare_latency_ms + execution_finalize_latency_ms
        self.last_effective_request = pending.arbitration.effective_request
        self.latest_execution_state = dict(timeline_record["execution_state"])
        self._pending_tick = None
        return {
            "timeline_record": timeline_record,
            "execution_state": dict(timeline_record["execution_state"]),
            "execution_latency_ms": float(total_execution_ms),
            "planner_prepare_latency_ms": float(pending.prepare_latency_ms),
            "execution_finalize_latency_ms": float(execution_finalize_latency_ms),
            "planner_execution_latency_ms": float(total_execution_ms),
            "arbitration_action": pending.arbitration.action,
            "arbitration_notes": list(pending.arbitration.notes),
        }

    def close(self) -> None:
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
