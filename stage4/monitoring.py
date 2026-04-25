from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .state_store import BehaviorFrameState, OnlineStackState, OnlineTickRecord


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


class Stage4Monitor:
    def __init__(self, *, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.tick_record_path = self.output_dir / "online_tick_record.jsonl"
        self.stack_state_path = self.output_dir / "online_stack_state.json"
        self.behavior_updates_path = self.output_dir / "world_behavior_updates.jsonl"
        self.latency_path = self.output_dir / "planner_control_latency.jsonl"
        self.fallback_events_path = self.output_dir / "fallback_events.jsonl"
        self.tick_record_path.unlink(missing_ok=True)
        self.behavior_updates_path.unlink(missing_ok=True)
        self.latency_path.unlink(missing_ok=True)
        self.fallback_events_path.unlink(missing_ok=True)
        self.tick_records: List[Dict[str, Any]] = []
        self.behavior_updates: List[Dict[str, Any]] = []
        self.latency_records: List[Dict[str, Any]] = []
        self.fallback_events: List[Dict[str, Any]] = []

    def record_behavior_update(self, behavior_state: BehaviorFrameState) -> None:
        payload = behavior_state.to_dict()
        self.behavior_updates.append(payload)
        _append_jsonl(self.behavior_updates_path, payload)

    def record_tick(self, tick_record: OnlineTickRecord, stack_state: OnlineStackState) -> None:
        tick_payload = tick_record.to_dict()
        stack_payload = stack_state.to_dict()
        self.tick_records.append(tick_payload)
        _append_jsonl(self.tick_record_path, tick_payload)
        _write_json(self.stack_state_path, stack_payload)
        latency_payload = {
            "tick_id": tick_payload.get("tick_id"),
            "request_frame": tick_payload.get("behavior_frame_id"),
            "sample_name": None,
            "carla_frame": tick_payload.get("frame_id"),
            "timestamp": tick_payload.get("timestamp"),
            "source_stage": "stage4_online",
            "measurement_mode": "direct_online_tick_contract",
            "perception_latency_ms": tick_payload.get("perception_latency_ms"),
            "world_update_latency_ms": tick_payload.get("world_update_latency_ms"),
            "behavior_latency_ms": tick_payload.get("behavior_latency_ms"),
            "planner_prepare_latency_ms": tick_payload.get("planner_prepare_latency_ms"),
            "execution_finalize_latency_ms": tick_payload.get("execution_finalize_latency_ms"),
            "planner_execution_latency_ms": tick_payload.get("planner_execution_latency_ms"),
            "decision_to_execution_latency_ms": tick_payload.get("decision_to_execution_latency_ms"),
            "total_loop_latency_ms": tick_payload.get("total_loop_latency_ms"),
            "latency_budget_ms": tick_payload.get("latency_budget_ms"),
            "over_budget": tick_payload.get("over_budget"),
            "over_budget_component": "total_loop_latency_ms" if tick_payload.get("over_budget") else None,
            "provenance": {
                "source_artifact": "online_tick_record.jsonl",
                "source_stage": "stage4",
            },
        }
        self.latency_records.append(latency_payload)
        _append_jsonl(self.latency_path, latency_payload)
        fallback_payload = {
            "tick_id": tick_payload.get("tick_id"),
            "request_frame": tick_payload.get("behavior_frame_id"),
            "sample_name": None,
            "carla_frame": tick_payload.get("frame_id"),
            "timestamp": tick_payload.get("timestamp"),
            "requested_behavior": tick_payload.get("selected_behavior"),
            "executing_behavior": None,
            "planner_status": tick_payload.get("execution_status"),
            "fallback_required": tick_payload.get("fallback_required"),
            "fallback_activated": tick_payload.get("fallback_activated"),
            "fallback_reason": tick_payload.get("fallback_reason"),
            "fallback_source": tick_payload.get("fallback_source"),
            "fallback_mode": tick_payload.get("fallback_mode"),
            "fallback_action": tick_payload.get("fallback_action"),
            "takeover_latency_ms": tick_payload.get("fallback_takeover_latency_ms"),
            "fallback_success": tick_payload.get("fallback_success"),
            "missing_when_required": bool(tick_payload.get("fallback_required") and not tick_payload.get("fallback_activated")),
            "notes": list(tick_payload.get("notes") or []),
        }
        self.fallback_events.append(fallback_payload)
        _append_jsonl(self.fallback_events_path, fallback_payload)
