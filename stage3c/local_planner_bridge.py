from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .execution_contract import ExecutionRequest
from .lane_change_executor import LaneChangeTracker
from stage3c_coverage.lane_change_completion import LaneChangeCompletionCriteria
from stage3c_coverage.stop_controller_tuner import StopControllerConfig, compute_stop_command


def ensure_carla_pythonapi(pythonapi_root: str | Path) -> None:
    pythonapi_root = Path(os.environ.get("CARLA_PYTHONAPI_ROOT", pythonapi_root))
    initial_carla_ready = False
    try:
        import carla  # type: ignore

        if hasattr(carla, "Client"):
            initial_carla_ready = True
    except ImportError:
        pass

    dist_dir = pythonapi_root / "carla" / "dist"
    version_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if dist_dir.exists():
        for wheel_path in dist_dir.glob(f"*{version_tag}*.whl"):
            path_str = str(wheel_path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
        for egg_path in dist_dir.glob(f"*{version_tag}*.egg"):
            path_str = str(egg_path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    supplemental_paths = [pythonapi_root / "carla", pythonapi_root]
    for path in supplemental_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.append(path_str)

    if not initial_carla_ready:
        sys.modules.pop("carla", None)

    try:
        import carla  # type: ignore

        if hasattr(carla, "Client"):
            return
        module_file = getattr(carla, "__file__", None)
        raise ModuleNotFoundError(
            "Imported 'carla' module is not the CARLA runtime API (missing Client). "
            f"module_file={module_file} python_tag={version_tag}"
        )
    except ImportError as exc:
        available = []
        if dist_dir.exists():
            available = [item.name for item in sorted(dist_dir.iterdir()) if item.is_file()]
        raise ModuleNotFoundError(
            "Unable to import 'carla' after PythonAPI bootstrap. "
            f"python_executable={sys.executable} python_tag={version_tag} "
            f"pythonapi_root={pythonapi_root} available_dist={available}"
        ) from exc


@dataclass
class PlannerStepResult:
    executed_behavior: str
    execution_status: str
    control: Any
    current_lane_id: int | None
    target_lane_id: int | None
    notes: List[str]


class LocalPlannerBridge:
    def __init__(
        self,
        *,
        vehicle: Any,
        map_inst: Any,
        pythonapi_root: str | Path,
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
    ) -> None:
        ensure_carla_pythonapi(pythonapi_root)
        import carla  # type: ignore
        from agents.navigation.local_planner import (  # type: ignore
            LocalPlanner,
            RoadOption,
            _compute_connection,
        )

        self._carla = carla
        self._LocalPlanner = LocalPlanner
        self._RoadOption = RoadOption
        self._compute_connection = _compute_connection

        self._vehicle = vehicle
        self._map = map_inst
        self._sampling_radius_m = float(sampling_radius_m)
        self._prepare_offset_m = float(prepare_offset_m)
        self._lane_change_completion_criteria = LaneChangeCompletionCriteria(
            success_hold_ticks=int(lane_change_success_hold_ticks),
            max_duration_ticks=int(max_lane_change_ticks),
            min_lateral_shift_m=float(lane_change_min_lateral_shift_m),
            min_lateral_shift_fraction=float(lane_change_min_lateral_shift_fraction),
        )
        self._stop_controller_config = StopControllerConfig(
            hold_ticks=int(stop_hold_ticks),
            full_stop_speed_threshold_mps=float(stop_full_stop_speed_threshold_mps),
            near_distance_m=float(stop_near_distance_m),
            hard_brake_distance_m=float(stop_hard_brake_distance_m),
        )
        self._stop_hold_ticks = int(stop_hold_ticks)
        self._active_lane_change: LaneChangeTracker | None = None
        self._active_stop_ticks_remaining = 0

        lateral_dict = {"K_P": 1.7, "K_I": 0.05, "K_D": 0.2, "dt": float(fixed_delta_seconds)}
        longitudinal_dict = {"K_P": 1.0, "K_I": 0.05, "K_D": 0.0, "dt": float(fixed_delta_seconds)}
        self._planner = self._LocalPlanner(
            self._vehicle,
            opt_dict={
                "dt": float(fixed_delta_seconds),
                "target_speed": 20.0,
                "sampling_radius": float(sampling_radius_m),
                "lateral_control_dict": lateral_dict,
                "longitudinal_control_dict": longitudinal_dict,
                "max_throttle": 0.65,
                "max_brake": 0.8,
                "max_steering": 0.8,
                "offset": 0.0,
                "base_min_distance": 2.5,
                "distance_ratio": 0.35,
            },
            map_inst=self._map,
        )

    @property
    def active_lane_change(self) -> LaneChangeTracker | None:
        return self._active_lane_change

    def post_tick_update(self, snapshot: Any) -> LaneChangeTracker | None:
        tracker = self._active_lane_change
        if tracker is None:
            return None
        tracker.update(vehicle=self._vehicle, map_inst=self._map, snapshot=snapshot)
        if tracker.status != "executing":
            self._active_lane_change = None
            return tracker
        return None

    def step(self, request: ExecutionRequest, snapshot: Any) -> PlannerStepResult:
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        current_lane_id = None if current_waypoint is None else int(current_waypoint.lane_id)
        notes: List[str] = []
        obstacle_distance_m = None

        if current_waypoint is None:
            control = self._stop_control(distance_to_obstacle_m=obstacle_distance_m)
            return PlannerStepResult(
                executed_behavior="stop_before_obstacle",
                execution_status="fail",
                control=control,
                current_lane_id=None,
                target_lane_id=None,
                notes=["current_waypoint_not_found"],
            )

        if self._active_stop_ticks_remaining > 0:
            self._active_stop_ticks_remaining -= 1
            control = self._stop_control(distance_to_obstacle_m=obstacle_distance_m)
            speed_mps = self._vehicle_speed_mps()
            status = (
                "success"
                if speed_mps <= self._stop_controller_config.full_stop_speed_threshold_mps
                else "executing"
            )
            notes.append("stop_hold_active")
            return PlannerStepResult(
                executed_behavior="stop_before_obstacle",
                execution_status=status,
                control=control,
                current_lane_id=current_lane_id,
                target_lane_id=current_lane_id,
                notes=notes,
            )

        if self._active_lane_change is not None and self._active_lane_change.status == "executing":
            active_direction = self._active_lane_change.direction
            active_behavior = f"commit_lane_change_{active_direction}"
            if request.requested_behavior == "stop_before_obstacle":
                self._active_lane_change.abort("superseded_by_stop_request")
                self._active_stop_ticks_remaining = max(self._stop_hold_ticks - 1, 0)
                control = self._stop_control(distance_to_obstacle_m=obstacle_distance_m)
                return PlannerStepResult(
                    executed_behavior="stop_before_obstacle",
                    execution_status="aborted",
                    control=control,
                    current_lane_id=current_lane_id,
                    target_lane_id=self._active_lane_change.target_lane_id,
                    notes=["active_lane_change_aborted_for_stop"],
                )
            self._planner.set_offset(0.0)
            self._planner.set_speed(self._mps_to_kmh(max(request.target_speed_mps, 4.0)))
            if self._planner.done():
                if current_lane_id == self._active_lane_change.target_lane_id:
                    plan = self._build_forward_plan(
                        current_waypoint,
                        route_option=request.route_option,
                        horizon_m=max(16.0, request.target_speed_mps * 3.0),
                    )
                else:
                    plan = self._build_lane_change_plan(
                        current_waypoint,
                        direction=active_direction,
                        horizon_m=max(24.0, request.target_speed_mps * 4.0),
                    )
                    if not plan:
                        plan = self._build_forward_plan(
                            current_waypoint,
                            route_option=request.route_option,
                            horizon_m=max(16.0, request.target_speed_mps * 3.0),
                        )
                        notes.append("active_lane_change_plan_regen_fallback")
                self._planner.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)
            control = self._planner.run_step()
            notes.append("continuing_active_lane_change")
            return PlannerStepResult(
                executed_behavior=active_behavior,
                execution_status="executing",
                control=control,
                current_lane_id=current_lane_id,
                target_lane_id=self._active_lane_change.target_lane_id,
                notes=notes,
            )

        behavior = request.requested_behavior
        if behavior == "stop_before_obstacle":
            self._active_stop_ticks_remaining = max(self._stop_hold_ticks - 1, 0)
            control = self._stop_control(distance_to_obstacle_m=obstacle_distance_m)
            speed_mps = self._vehicle_speed_mps()
            status = (
                "success"
                if speed_mps <= self._stop_controller_config.full_stop_speed_threshold_mps
                else "executing"
            )
            return PlannerStepResult(
                executed_behavior=behavior,
                execution_status=status,
                control=control,
                current_lane_id=current_lane_id,
                target_lane_id=current_lane_id,
                notes=notes,
            )

        target_lane_id = current_lane_id
        if behavior.startswith("prepare_lane_change_"):
            direction = "right" if behavior.endswith("right") else "left"
            target_lane_id = self._adjacent_lane_id(current_waypoint, direction)
            offset = self._prepare_offset(direction)
            self._planner.set_offset(offset)
            plan = self._build_forward_plan(
                current_waypoint,
                route_option=request.route_option,
                horizon_m=max(18.0, request.target_speed_mps * 3.5),
            )
            notes.append(f"prepare_offset_{direction}")
        elif behavior.startswith("commit_lane_change_"):
            direction = "right" if behavior.endswith("right") else "left"
            target_lane_id = self._adjacent_lane_id(current_waypoint, direction)
            plan = self._build_lane_change_plan(
                current_waypoint,
                direction=direction,
                horizon_m=max(24.0, request.target_speed_mps * 4.0),
            )
            self._planner.set_offset(0.0)
            if not plan or target_lane_id is None:
                notes.append("lane_change_plan_unavailable")
                plan = self._build_forward_plan(
                    current_waypoint,
                    route_option=request.route_option,
                    horizon_m=max(16.0, request.target_speed_mps * 3.0),
                )
                self._planner.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)
                self._planner.set_speed(self._mps_to_kmh(request.target_speed_mps))
                control = self._planner.run_step()
                return PlannerStepResult(
                    executed_behavior="keep_lane",
                    execution_status="fail",
                    control=control,
                    current_lane_id=current_lane_id,
                    target_lane_id=target_lane_id,
                    notes=notes,
                )
            self._active_lane_change = self._start_lane_change_tracker(
                request=request,
                snapshot=snapshot,
                current_waypoint=current_waypoint,
                target_lane_id=target_lane_id,
                direction=direction,
            )
        else:
            self._planner.set_offset(0.0)
            plan = self._build_forward_plan(
                current_waypoint,
                route_option=request.route_option,
                horizon_m=max(16.0, request.target_speed_mps * 3.0),
            )
            if behavior == "keep_route_through_junction":
                notes.append("route_conditioned_junction_follow")

        self._planner.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)
        self._planner.set_speed(self._mps_to_kmh(request.target_speed_mps))
        control = self._planner.run_step()
        return PlannerStepResult(
            executed_behavior=behavior,
            execution_status="executing",
            control=control,
            current_lane_id=current_lane_id,
            target_lane_id=target_lane_id,
            notes=notes,
        )

    def _build_forward_plan(self, start_waypoint: Any, *, route_option: str, horizon_m: float) -> List[Any]:
        route_preference = self._resolve_route_preference(route_option)
        current = start_waypoint
        plan: List[Any] = [(current, self._RoadOption.LANEFOLLOW)]
        distance = 0.0
        while distance < horizon_m:
            next_waypoint, road_option = self._next_waypoint(current, route_preference)
            if next_waypoint is None:
                break
            distance += float(next_waypoint.transform.location.distance(current.transform.location))
            plan.append((next_waypoint, road_option))
            current = next_waypoint
        return plan

    def _build_lane_change_plan(self, start_waypoint: Any, *, direction: str, horizon_m: float) -> List[Any]:
        same_lane_distance = min(8.0, max(4.0, horizon_m * 0.25))
        lane_change_distance = min(10.0, max(5.0, horizon_m * 0.20))
        other_lane_distance = max(12.0, horizon_m - same_lane_distance)
        plan: List[Any] = [(start_waypoint, self._RoadOption.LANEFOLLOW)]

        current = start_waypoint
        distance = 0.0
        while distance < same_lane_distance:
            next_candidates = list(current.next(self._sampling_radius_m))
            if not next_candidates:
                return []
            next_waypoint = next_candidates[0]
            distance += float(next_waypoint.transform.location.distance(current.transform.location))
            plan.append((next_waypoint, self._RoadOption.LANEFOLLOW))
            current = next_waypoint

        lane_change_anchor_candidates = list(current.next(max(lane_change_distance, self._sampling_radius_m)))
        if not lane_change_anchor_candidates:
            return []
        lane_change_anchor = lane_change_anchor_candidates[0]
        if direction == "right":
            side_waypoint = lane_change_anchor.get_right_lane()
            lane_change_option = self._RoadOption.CHANGELANERIGHT
            allowed = str(lane_change_anchor.lane_change) in {"Right", "Both"}
        else:
            side_waypoint = lane_change_anchor.get_left_lane()
            lane_change_option = self._RoadOption.CHANGELANELEFT
            allowed = str(lane_change_anchor.lane_change) in {"Left", "Both"}
        if not allowed or side_waypoint is None or side_waypoint.lane_type != self._carla.LaneType.Driving:
            return []
        plan.append((side_waypoint, lane_change_option))

        current = side_waypoint
        distance = 0.0
        while distance < other_lane_distance:
            next_candidates = list(current.next(self._sampling_radius_m))
            if not next_candidates:
                break
            next_waypoint = next_candidates[0]
            distance += float(next_waypoint.transform.location.distance(current.transform.location))
            plan.append((next_waypoint, self._RoadOption.LANEFOLLOW))
            current = next_waypoint
        return plan

    def _next_waypoint(self, current_waypoint: Any, route_preference: str) -> tuple[Any | None, Any]:
        next_waypoints = list(current_waypoint.next(self._sampling_radius_m))
        if not next_waypoints:
            return None, self._RoadOption.LANEFOLLOW
        if len(next_waypoints) == 1:
            return next_waypoints[0], self._RoadOption.LANEFOLLOW

        preferred_option = self._route_preference_to_road_option(route_preference)
        candidates: List[tuple[Any, Any]] = []
        for candidate in next_waypoints:
            next_next = list(candidate.next(3.0))
            comparison_waypoint = next_next[0] if next_next else candidate
            road_option = self._compute_connection(current_waypoint, comparison_waypoint)
            candidates.append((candidate, road_option))
        if preferred_option is not None:
            for candidate, road_option in candidates:
                if road_option == preferred_option:
                    return candidate, road_option
        return candidates[0]

    def _resolve_route_preference(self, route_option: str) -> str:
        value = str(route_option or "lane_follow")
        if value in {"left", "keep_left", "prepare_turn_left"}:
            return "left"
        if value in {"right", "keep_right", "prepare_turn_right"}:
            return "right"
        if value in {"straight", "straight_through_junction"}:
            return "straight"
        return "lane_follow"

    def _route_preference_to_road_option(self, route_preference: str) -> Any | None:
        if route_preference == "left":
            return self._RoadOption.LEFT
        if route_preference == "right":
            return self._RoadOption.RIGHT
        if route_preference == "straight":
            return self._RoadOption.STRAIGHT
        return None

    def _prepare_offset(self, direction: str) -> float:
        lane_width = self._map.get_waypoint(self._vehicle.get_location()).lane_width
        bounded = min(abs(self._prepare_offset_m), max(0.3, lane_width * 0.33))
        return bounded if direction == "right" else -bounded

    def _adjacent_lane_id(self, waypoint: Any, direction: str) -> int | None:
        if direction == "right":
            side_waypoint = waypoint.get_right_lane()
        else:
            side_waypoint = waypoint.get_left_lane()
        if side_waypoint is None or side_waypoint.lane_type != self._carla.LaneType.Driving:
            return None
        return int(side_waypoint.lane_id)

    def _start_lane_change_tracker(
        self,
        *,
        request: ExecutionRequest,
        snapshot: Any,
        current_waypoint: Any,
        target_lane_id: int,
        direction: str,
    ) -> LaneChangeTracker:
        location = self._vehicle.get_location()
        right_vector = self._vehicle.get_transform().get_right_vector()
        return LaneChangeTracker(
            request_frame=int(request.frame),
            sample_name=str(request.sample_name),
            requested_behavior=str(request.requested_behavior),
            direction=direction,
            start_carla_frame=int(snapshot.frame),
            start_timestamp=float(snapshot.timestamp.elapsed_seconds),
            start_lane_id=int(current_waypoint.lane_id),
            target_lane_id=int(target_lane_id),
            lane_width_m=float(current_waypoint.lane_width),
            start_location_xyz=[float(location.x), float(location.y), float(location.z)],
            start_right_vector_xyz=[float(right_vector.x), float(right_vector.y), float(right_vector.z)],
            completion_criteria=self._lane_change_completion_criteria,
        )

    def _vehicle_speed_mps(self) -> float:
        velocity = self._vehicle.get_velocity()
        return math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)

    def _mps_to_kmh(self, speed_mps: float) -> float:
        return max(0.0, float(speed_mps) * 3.6)

    def _stop_control(self, *, distance_to_obstacle_m: float | None) -> Any:
        stop_command = compute_stop_command(
            speed_mps=self._vehicle_speed_mps(),
            distance_to_obstacle_m=distance_to_obstacle_m,
            config=self._stop_controller_config,
        )
        control = self._carla.VehicleControl()
        control.throttle = float(stop_command["throttle"])
        control.steer = float(stop_command["steer"])
        control.brake = float(stop_command["brake"])
        control.hand_brake = bool(stop_command["hand_brake"])
        control.manual_gear_shift = bool(stop_command["manual_gear_shift"])
        return control
