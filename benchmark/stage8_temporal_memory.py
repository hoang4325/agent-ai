"""
Stage 8 — Temporal Memory Store
=================================
Maintains a sliding window of the last N agent BRP frames.
Used by the Arbitration Engine to detect intent thrashing,
track active maneuver state, and enforce temporal consistency.

INVARIANTS:
  - Memory is NOT injected into the LLM prompt.
  - Memory does NOT have control authority.
  - Memory is wiped on case boundary / stop completion / emergency brake.
  - Agent must re-infer context from raw state each frame.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Intents that form a lane-change maneuver sequence
_LC_PREPARE_INTENTS = frozenset({
    "prepare_lane_change_left",
    "prepare_lane_change_right",
})
_LC_COMMIT_INTENTS = frozenset({
    "commit_lane_change_left",
    "commit_lane_change_right",
})
_LC_ABORT_INTENTS = frozenset({"abort_lane_change"})
_STOP_INTENTS = frozenset({"stop"})


@dataclass
class FrameMemoryEntry:
    frame_id: int
    agent_intent: str
    approved: bool
    veto_code: str | None
    confidence: float
    agrees_with_baseline: bool
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class TemporalMemoryState:
    memory_id: str
    case_id: str
    window_size: int
    frames: list[dict[str, Any]]         # most-recent N FrameMemoryEntry dicts
    active_maneuver: str | None          # e.g. "lane_change_right"
    maneuver_frame_start: int | None
    maneuver_committed: bool
    reset_count: int
    total_frames_seen: int


class TemporalMemoryStore:
    """
    Sliding window memory for Stage 8 Assist.

    AE reads this before deciding to approve/veto a BRP.
    Memory is updated AFTER AE decision (approved or not).
    """

    def __init__(self, case_id: str, window_size: int = 5):
        import uuid
        self._case_id = case_id
        self._window_size = window_size
        self._frames: list[FrameMemoryEntry] = []
        self._active_maneuver: str | None = None
        self._maneuver_frame_start: int | None = None
        self._maneuver_committed: bool = False
        self._reset_count: int = 0
        self._total_frames_seen: int = 0
        self._memory_id: str = str(uuid.uuid4())

    # ── Read interface (for AE) ────────────────────────────────────────────────

    @property
    def active_maneuver(self) -> str | None:
        return self._active_maneuver

    @property
    def maneuver_committed(self) -> bool:
        return self._maneuver_committed

    @property
    def window_size(self) -> int:
        return self._window_size

    def last_n_intents(self, n: int | None = None) -> list[str]:
        """Return list of agent_intent strings from the last N approved frames."""
        n = n or self._window_size
        return [
            e.agent_intent for e in self._frames[-n:]
        ]

    def last_approved_intent(self) -> str | None:
        """Most recent approved intent, or None."""
        for entry in reversed(self._frames):
            if entry.approved:
                return entry.agent_intent
        return None

    def detect_temporal_conflict(self, candidate_intent: str) -> bool:
        """
        Returns True if candidate_intent conflicts with the active maneuver state.

        Conflict cases:
          - Maneuver is committed (e.g. commit_lc_right in progress) and
            candidate is contradictory (e.g. keep_lane / prepare other side)
            without first issuing an abort.
        """
        if not self._maneuver_committed:
            return False
        if not self._active_maneuver:
            return False

        # Determine committed direction
        committed_right = "right" in (self._active_maneuver or "")
        committed_left = "left" in (self._active_maneuver or "")

        candidate_is_abort = candidate_intent in _LC_ABORT_INTENTS
        candidate_is_commit_same = (
            (committed_right and candidate_intent == "commit_lane_change_right") or
            (committed_left and candidate_intent == "commit_lane_change_left")
        )
        candidate_is_continue = candidate_is_commit_same or candidate_is_abort

        # Contradiction: maneuver committed but agent proposes something else
        return not candidate_is_continue

    def snapshot(self) -> TemporalMemoryState:
        return TemporalMemoryState(
            memory_id=self._memory_id,
            case_id=self._case_id,
            window_size=self._window_size,
            frames=[asdict(e) for e in self._frames],
            active_maneuver=self._active_maneuver,
            maneuver_frame_start=self._maneuver_frame_start,
            maneuver_committed=self._maneuver_committed,
            reset_count=self._reset_count,
            total_frames_seen=self._total_frames_seen,
        )

    # ── Write interface (called by AE after decision) ─────────────────────────

    def record(
        self,
        frame_id: int,
        agent_intent: str,
        approved: bool,
        veto_code: str | None,
        confidence: float,
        agrees_with_baseline: bool,
    ) -> None:
        entry = FrameMemoryEntry(
            frame_id=frame_id,
            agent_intent=agent_intent,
            approved=approved,
            veto_code=veto_code,
            confidence=confidence,
            agrees_with_baseline=agrees_with_baseline,
        )
        self._frames.append(entry)
        # Keep only the last window_size entries
        if len(self._frames) > self._window_size:
            self._frames = self._frames[-self._window_size:]

        self._total_frames_seen += 1

        # Update active maneuver state (only on approved frames)
        if approved:
            self._update_maneuver_state(frame_id, agent_intent)

    def reset(self, reason: str = "explicit") -> None:
        """Wipe memory — call on case boundary, stop completion, emergency brake."""
        self._frames = []
        self._active_maneuver = None
        self._maneuver_frame_start = None
        self._maneuver_committed = False
        self._reset_count += 1

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update_maneuver_state(self, frame_id: int, intent: str) -> None:
        if intent in _LC_PREPARE_INTENTS:
            # Start a new maneuver prepare phase
            direction = "right" if "right" in intent else "left"
            self._active_maneuver = f"lane_change_{direction}"
            self._maneuver_frame_start = frame_id
            self._maneuver_committed = False

        elif intent in _LC_COMMIT_INTENTS:
            # Mark maneuver as committed
            direction = "right" if "right" in intent else "left"
            if not self._active_maneuver:
                self._active_maneuver = f"lane_change_{direction}"
                self._maneuver_frame_start = frame_id
            self._maneuver_committed = True

        elif intent in _LC_ABORT_INTENTS:
            # Abort clears maneuver
            self._active_maneuver = None
            self._maneuver_frame_start = None
            self._maneuver_committed = False

        elif intent in _STOP_INTENTS:
            # Stop clears any ongoing LC maneuver
            self._active_maneuver = None
            self._maneuver_frame_start = None
            self._maneuver_committed = False

        elif intent == "keep_lane":
            # If we were in a non-committed prepare, cancel it
            if self._active_maneuver and not self._maneuver_committed:
                self._active_maneuver = None
                self._maneuver_frame_start = None

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        snap = self.snapshot()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(snap), f, indent=2)
