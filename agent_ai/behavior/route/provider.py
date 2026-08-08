from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


_ROUTE_OPTION_ALIASES = {
    "follow": "lane_follow",
    "lane_follow": "lane_follow",
    "keep_lane": "lane_follow",
    "left": "left",
    "turn_left": "left",
    "prepare_turn_left": "left",
    "right": "right",
    "turn_right": "right",
    "prepare_turn_right": "right",
    "straight": "straight",
    "straight_through_junction": "straight",
    "keep_left": "keep_left",
    "keep_right": "keep_right",
    "unknown": "unknown",
}

_PREFERRED_LANE_ALIASES = {
    "current": "current",
    "keep_current": "current",
    "center": "current",
    "left": "left",
    "prefer_left": "left",
    "right": "right",
    "prefer_right": "right",
    "unknown": "unknown",
}


def normalize_route_option(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _ROUTE_OPTION_ALIASES.get(str(value).strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported route option: {value}")
    return normalized


def normalize_preferred_lane(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _PREFERRED_LANE_ALIASES.get(str(value).strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported preferred lane: {value}")
    return normalized


def preferred_lane_from_route_option(route_option: str | None) -> str | None:
    if route_option is None:
        return None
    if route_option in {"keep_left", "left"}:
        return "left"
    if route_option in {"keep_right", "right"}:
        return "right"
    if route_option in {"lane_follow", "straight"}:
        return "current"
    return "unknown"


def load_route_manifest(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Route manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Route manifest must be a JSON object: {manifest_path}")
    return payload


def _override_for_record(route_manifest: Dict[str, Any], stage3_record: Dict[str, Any]) -> Dict[str, Any]:
    sample_name = str(stage3_record["sample_name"])
    frame_id = int(stage3_record["frame_id"])
    sample_overrides = route_manifest.get("sample_overrides", {})
    frame_overrides = route_manifest.get("frame_overrides", {})
    override: Dict[str, Any] = {}
    if isinstance(sample_overrides, dict) and sample_name in sample_overrides:
        sample_payload = sample_overrides[sample_name]
        if isinstance(sample_payload, dict):
            override.update(sample_payload)
    if isinstance(frame_overrides, dict) and str(frame_id) in frame_overrides:
        frame_payload = frame_overrides[str(frame_id)]
        if isinstance(frame_payload, dict):
            override.update(frame_payload)
    return override


def resolve_route_hint(
    *,
    stage3_record: Dict[str, Any],
    cli_route_option: str | None,
    cli_preferred_lane: str | None,
    cli_route_priority: str | None,
    route_manifest: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    route_manifest = route_manifest or {}
    junction_context = stage3_record["lane_context"]["junction_context"]
    manifest_override = _override_for_record(route_manifest, stage3_record)

    manifest_route_option = normalize_route_option(route_manifest.get("default_route_option"))
    manifest_preferred_lane = normalize_preferred_lane(route_manifest.get("default_preferred_lane"))
    manifest_route_priority = route_manifest.get("default_route_priority")
    manifest_route_mode = str(route_manifest.get("route_mode", "replay_route_manifest"))

    override_route_option = normalize_route_option(manifest_override.get("route_option"))
    override_preferred_lane = normalize_preferred_lane(manifest_override.get("preferred_lane"))
    override_route_priority = manifest_override.get("route_priority")

    cli_route_option = normalize_route_option(cli_route_option)
    cli_preferred_lane = normalize_preferred_lane(cli_preferred_lane)
    cli_route_priority = None if cli_route_priority is None else str(cli_route_priority).strip().lower()

    route_conflict_flags: list[str] = []
    if override_route_option or override_preferred_lane or override_route_priority:
        route_mode = manifest_override.get("route_mode", manifest_route_mode)
        route_option = override_route_option or manifest_route_option
        preferred_lane = override_preferred_lane or manifest_preferred_lane
        route_priority = str(override_route_priority or manifest_route_priority or "normal")
        hint_source = "route_manifest_override"
    elif cli_route_option or cli_preferred_lane:
        route_mode = "minimal_route_hint"
        route_option = cli_route_option
        preferred_lane = cli_preferred_lane
        route_priority = str(cli_route_priority or "normal")
        hint_source = "cli_route_hint"
    elif manifest_route_option or manifest_preferred_lane:
        route_mode = manifest_route_mode
        route_option = manifest_route_option
        preferred_lane = manifest_preferred_lane
        route_priority = str(manifest_route_priority or "normal")
        hint_source = "route_manifest_default"
    else:
        route_mode = "corridor_only"
        if bool(junction_context.get("is_in_junction")) or bool(junction_context.get("junction_ahead")):
            if int(junction_context.get("branch_count_ahead", 0)) > 1:
                route_option = "unknown"
                preferred_lane = "unknown"
                route_conflict_flags.append("route_decision_not_provided")
            else:
                route_option = "straight"
                preferred_lane = "current"
        else:
            route_option = "lane_follow"
            preferred_lane = "current"
        route_priority = "normal"
        hint_source = "corridor_inference"

    route_option = route_option or "unknown"
    preferred_lane = preferred_lane or preferred_lane_from_route_option(route_option) or "unknown"
    if preferred_lane == "unknown" and route_option in {"lane_follow", "straight"}:
        preferred_lane = "current"
    if preferred_lane == "unknown" and route_option in {"keep_left", "left"}:
        preferred_lane = "left"
    if preferred_lane == "unknown" and route_option in {"keep_right", "right"}:
        preferred_lane = "right"

    return {
        "route_mode": str(route_mode),
        "route_option": str(route_option),
        "preferred_lane": str(preferred_lane),
        "route_priority": str(route_priority),
        "hint_source": str(hint_source),
        "route_conflict_flags": route_conflict_flags,
    }
