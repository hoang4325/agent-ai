"""Latest-only background worker for slow tactical Agent calls.

The worker deliberately keeps the LLM outside the CARLA control loop.  At most
one request is executing and at most one newer request is pending; submitting a
new state replaces the pending (not yet started) state.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class AsyncAgentRequest:
    request_id: int
    frame_id: int
    frame_idx: int
    sim_timestamp_s: float
    submitted_wall_s: float
    payload: Dict[str, Any]
    context: Dict[str, Any]


@dataclass(frozen=True)
class AsyncAgentResult:
    request: AsyncAgentRequest
    intent_record: Any
    completed_wall_s: float
    latency_ms: float
    error_type: Optional[str] = None


@dataclass(frozen=True)
class AsyncSubmitOutcome:
    request_id: int
    replaced_pending_request_id: Optional[int]


class LatestOnlyAgentWorker:
    """Run a blocking Agent callable on one daemon thread without queue growth."""

    def __init__(
        self,
        call: Callable[[Dict[str, Any]], Any],
        *,
        name: str = "agent-intent-worker",
    ) -> None:
        self._call = call
        self._condition = threading.Condition()
        self._pending: Optional[AsyncAgentRequest] = None
        self._inflight: Optional[AsyncAgentRequest] = None
        self._latest_result: Optional[AsyncAgentResult] = None
        self._stopping = False
        self._next_request_id = 1
        self._submitted = 0
        self._started = 0
        self._completed = 0
        self._pending_replaced = 0
        self._unconsumed_result_replaced = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(
        self,
        *,
        frame_id: int,
        frame_idx: int,
        sim_timestamp_s: float,
        payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncSubmitOutcome:
        with self._condition:
            if self._stopping:
                raise RuntimeError("Agent worker is stopping")
            request_id = self._next_request_id
            self._next_request_id += 1
            replaced_id = self._pending.request_id if self._pending is not None else None
            if replaced_id is not None:
                self._pending_replaced += 1
            self._pending = AsyncAgentRequest(
                request_id=request_id,
                frame_id=int(frame_id),
                frame_idx=int(frame_idx),
                sim_timestamp_s=float(sim_timestamp_s),
                submitted_wall_s=time.monotonic(),
                payload=dict(payload),
                context=dict(context or {}),
            )
            self._submitted += 1
            self._condition.notify()
            return AsyncSubmitOutcome(request_id, replaced_id)

    def poll(self) -> Optional[AsyncAgentResult]:
        with self._condition:
            result = self._latest_result
            self._latest_result = None
            return result

    def stats(self) -> Dict[str, Any]:
        with self._condition:
            return {
                "submitted": self._submitted,
                "started": self._started,
                "completed": self._completed,
                "pending_replaced": self._pending_replaced,
                "unconsumed_result_replaced": self._unconsumed_result_replaced,
                "inflight": self._inflight is not None,
                "pending": self._pending is not None,
            }

    def close(self, *, join_timeout_s: float = 0.25) -> None:
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, float(join_timeout_s)))

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                request = self._pending
                self._pending = None
                self._inflight = request
                self._started += 1

            assert request is not None
            intent_record: Any = None
            error_type: Optional[str] = None
            try:
                intent_record = self._call(request.payload)
            except Exception as exc:  # The control loop must survive provider failures.
                error_type = type(exc).__name__

            completed_wall_s = time.monotonic()
            result = AsyncAgentResult(
                request=request,
                intent_record=intent_record,
                completed_wall_s=completed_wall_s,
                latency_ms=max(0.0, (completed_wall_s - request.submitted_wall_s) * 1000.0),
                error_type=error_type,
            )
            with self._condition:
                self._inflight = None
                if self._latest_result is not None:
                    self._unconsumed_result_replaced += 1
                self._latest_result = result
                self._completed += 1
