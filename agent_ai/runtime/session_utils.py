"""
CARLA session helpers used by Stage 4 online orchestration.

Extracted so orchestrator stays focused on the tick loop and helpers can be
unit-tested without spinning up the full online stack.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from agent_ai.shared.artifact_io import load_json

LOGGER = logging.getLogger("stage4.session_utils")


def normalize_town_name(town: str) -> str:
    """Return the leaf town name from a path-like CARLA map string."""
    normalized = str(town).replace("\\", "/")
    return normalized.split("/")[-1]


def call_world_api_with_retry(
    *,
    action: Any,
    description: str,
    attempts: int = 6,
    sleep_seconds: float = 2.0,
    logger: logging.Logger | None = None,
) -> Any:
    """Retry a CARLA world API call that may raise RuntimeError while loading."""
    log = logger or LOGGER
    last_error: Exception | None = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            return action()
        except RuntimeError as exc:
            last_error = exc
            if attempt >= attempts:
                break
            log.warning(
                "Stage4 %s not ready on attempt %d/%d: %s",
                description,
                attempt,
                attempts,
                exc,
            )
            time.sleep(float(sleep_seconds))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Stage4 failed to resolve {description}.")


def resolve_attach_actor_id(args: Any) -> tuple[int | None, dict[str, Any] | None]:
    """
    Resolve ego attach target from CLI args.

    Returns ``(actor_id, scenario_manifest_payload_or_None)``.
    """
    if getattr(args, "attach_to_actor_id", None):
        return int(args.attach_to_actor_id), None
    if not getattr(args, "scenario_manifest", None):
        return None, None
    manifest_path = Path(args.scenario_manifest)
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Scenario manifest {manifest_path} must be a JSON object.")
    actor_id = manifest.get("ego_actor_id")
    if not isinstance(actor_id, int) or actor_id <= 0:
        raise ValueError(f"Scenario manifest {manifest_path} does not contain a valid ego_actor_id.")
    return int(actor_id), manifest


def clone_execution_request_with_updates(request: Any, **updates: Any) -> Any:
    """Return a new ExecutionRequest with selected fields overridden."""
    payload = request.to_dict()
    payload.update(updates)
    from agent_ai.behavior.execution.execution_contract import ExecutionRequest

    return ExecutionRequest(**payload)
