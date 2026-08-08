from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass
class TraceLogger:
    benchmark_run_id: str
    trace_file: Path
    contract_trace_file: Path | None = None

    def emit(
        self,
        *,
        case_id: str,
        trace_id: str,
        benchmark_mode: str,
        step: str,
        status: str,
        artifact_paths: list[str] | None = None,
        notes: list[str] | None = None,
        stage_origin: str | None = None,
        stage_instance_id: str | None = None,
        current_run_only: bool | None = None,
        timing_ms: float | None = None,
        failure_reason: str | None = None,
        orchestration_status: str | None = None,
    ) -> None:
        payload = {
            "benchmark_run_id": self.benchmark_run_id,
            "case_id": case_id,
            "trace_id": trace_id,
            "benchmark_mode": benchmark_mode,
            "step": step,
            "status": status,
            "stage_origin": stage_origin or step,
            "stage_instance_id": stage_instance_id or step,
            "current_run_only": current_run_only,
            "timing_ms": timing_ms,
            "failure_reason": failure_reason,
            "orchestration_status": orchestration_status or status,
            "artifact_paths": artifact_paths or [],
            "notes": notes or [],
            "timestamp": utc_now_iso(),
        }
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload))
            handle.write("\n")
        if self.contract_trace_file is not None:
            self.contract_trace_file.parent.mkdir(parents=True, exist_ok=True)
            with self.contract_trace_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload))
                handle.write("\n")


def write_artifact_provenance(
    *,
    output_path: Path,
    benchmark_run_id: str,
    case_id: str,
    trace_id: str,
    artifacts: list[dict[str, Any]],
) -> None:
    payload = {
        "benchmark_run_id": benchmark_run_id,
        "case_id": case_id,
        "trace_id": trace_id,
        "generated_at": utc_now_iso(),
        "artifacts": artifacts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
