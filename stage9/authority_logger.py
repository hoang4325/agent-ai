"""
Stage 9 — Authority Logger (v2)
=================================
Structured audit log for every authority event:
  grant, reject, revoke, TOR, MRM, SAFE_STOP, human_override

Each event is written to a JSONL file for offline analysis and scenario evaluation.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class AuthorityLogger:
    """
    Append-only structured logger for Stage 9 authority events.

    Events:
      GRANT, REJECT, REVOKE, TOR_START, TOR_TIMEOUT, MRM_START, SAFE_STOP,
      HUMAN_OVERRIDE, COOLDOWN_ACTIVE, VETO
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        if log_path is None:
            log_path = Path("benchmark/reports/stage9_authority_log.jsonl")
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._frame_id: int = 0

    def set_frame(self, frame_id: int) -> None:
        self._frame_id = frame_id

    # ── Event methods ─────────────────────────────────────────────────────────

    def log_grant(
        self,
        contract_id: str,
        tactical_intent: str,
        confidence: float,
        max_duration_s: float,
        state_from: str,
    ) -> None:
        self._write({
            "event": "GRANT",
            "contract_id": contract_id,
            "tactical_intent": tactical_intent,
            "confidence": round(confidence, 3),
            "max_duration_s": max_duration_s,
            "state_from": state_from,
        })

    def log_reject(
        self,
        primary_reason: str,
        all_reasons: list[str],
        contract_id: Optional[str] = None,
    ) -> None:
        self._write({
            "event": "REJECT",
            "contract_id": contract_id,
            "primary_reason": primary_reason,
            "all_reasons": all_reasons,
        })

    def log_revoke(
        self,
        reason: str,
        contract_id: Optional[str] = None,
        revoke_alpha_at_trigger: float = 1.0,
    ) -> None:
        self._write({
            "event": "REVOKE",
            "contract_id": contract_id,
            "reason": reason,
            "revoke_alpha_at_trigger": round(revoke_alpha_at_trigger, 3),
        })

    def log_veto(self, reason: str, severity: str, contract_id: Optional[str] = None) -> None:
        self._write({
            "event": "VETO",
            "contract_id": contract_id,
            "reason": reason,
            "severity": severity,
        })

    def log_tor_start(self, reason: str, time_budget_s: Optional[float]) -> None:
        self._write({
            "event": "TOR_START",
            "reason": reason,
            "time_budget_s": time_budget_s,
        })

    def log_tor_timeout(self) -> None:
        self._write({"event": "TOR_TIMEOUT"})

    def log_mrm_start(self, reason: str, strategy: str) -> None:
        self._write({
            "event": "MRM_START",
            "reason": reason,
            "mrm_strategy": strategy,
        })

    def log_safe_stop(self) -> None:
        self._write({"event": "SAFE_STOP"})

    def log_human_override(self) -> None:
        self._write({"event": "HUMAN_OVERRIDE"})

    def log_state_transition(self, from_state: str, to_state: str, reason: str) -> None:
        self._write({
            "event": "STATE_TRANSITION",
            "from": from_state,
            "to": to_state,
            "reason": reason,
        })

    def log_cooldown(self, remaining_s: float) -> None:
        self._write({
            "event": "COOLDOWN_ACTIVE",
            "remaining_s": round(remaining_s, 2),
        })

    # ── Internal ─────────────────────────────────────────────────────────────

    def _write(self, payload: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "frame_id": self._frame_id,
            **payload,
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
