from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmark.kinematic_mpc_shadow import (
    _build_shadow_summary_from_proposals,
    new_shadow_runtime_state,
    solve_kinematic_shadow_step,
)
from benchmark.shadow_artifacts import (
    _merge_realized_outcome_into_shadow_summary,
    build_shadow_baseline_comparison_payload,
    build_shadow_realized_outcome_payload,
)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")


def _touch_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


class Stage4ShadowRuntime:
    def __init__(self, *, output_dir: str | Path, case_id: str) -> None:
        self.output_dir = Path(output_dir)
        self.case_id = str(case_id)
        self.runtime_state = new_shadow_runtime_state()
        self.proposals: list[dict[str, Any]] = []
        self.disagreement_events: list[dict[str, Any]] = []
        self.proposal_path = self.output_dir / "mpc_shadow_proposal.jsonl"
        self.summary_path = self.output_dir / "mpc_shadow_summary.json"
        self.comparison_path = self.output_dir / "shadow_baseline_comparison.json"
        self.disagreement_path = self.output_dir / "shadow_disagreement_events.jsonl"
        self.realized_trace_path = self.output_dir / "shadow_realized_outcome_trace.jsonl"
        self.realized_summary_path = self.output_dir / "shadow_realized_outcome_summary.json"
        self.latency_trace_path = self.output_dir / "shadow_runtime_latency_trace.jsonl"
        self.latency_summary_path = self.output_dir / "shadow_runtime_latency_summary.json"
        self.latency_trace_rows: list[dict[str, Any]] = []
        self.timeline_records: list[dict[str, Any]] = []
        self.proposal_path.unlink(missing_ok=True)
        self.disagreement_path.unlink(missing_ok=True)
        self.realized_trace_path.unlink(missing_ok=True)
        self.latency_trace_path.unlink(missing_ok=True)
        _touch_jsonl(self.proposal_path)
        _touch_jsonl(self.disagreement_path)
        _touch_jsonl(self.realized_trace_path)
        _touch_jsonl(self.latency_trace_path)

    def record_tick(
        self,
        *,
        timeline_record: dict[str, Any],
        planner_execution_latency_ms: float | None,
    ) -> dict[str, Any]:
        started = perf_counter()
        self.timeline_records.append(dict(timeline_record))
        step = solve_kinematic_shadow_step(
            case_id=self.case_id,
            row=timeline_record,
            control_row=dict(timeline_record.get("applied_control") or {}),
            latency_row={"planner_execution_latency_ms": planner_execution_latency_ms},
            runtime_state=self.runtime_state,
            source_stage="stage4_online",
        )
        proposal = dict(step["proposal"])
        self.proposals.append(proposal)
        disagreement_event = step.get("disagreement_event")
        if disagreement_event:
            disagreement_payload = dict(disagreement_event)
            self.disagreement_events.append(disagreement_payload)
        total_runtime_ms = float((perf_counter() - started) * 1000.0)
        solver_latency_ms = float(proposal.get("shadow_planner_execution_latency_ms") or 0.0)
        build_latency_ms = float(proposal.get("shadow_planner_problem_build_latency_ms") or 0.0)
        setup_latency_ms = float(proposal.get("shadow_planner_setup_latency_ms") or 0.0)
        total_compute_latency_ms = float(proposal.get("shadow_planner_total_compute_latency_ms") or solver_latency_ms)
        bookkeeping_latency_ms = max(total_runtime_ms - total_compute_latency_ms, 0.0)
        self.latency_trace_rows.append(
            {
                "case_id": self.case_id,
                "request_frame": proposal.get("request_frame"),
                "carla_frame": proposal.get("carla_frame"),
                "sample_name": proposal.get("sample_name"),
                "solver_latency_ms": solver_latency_ms,
                "problem_build_latency_ms": build_latency_ms,
                "solver_setup_latency_ms": setup_latency_ms,
                "planner_total_compute_latency_ms": total_compute_latency_ms,
                "shadow_tick_runtime_ms": total_runtime_ms,
                "bookkeeping_latency_ms": bookkeeping_latency_ms,
                "proposal_feasible": bool(proposal.get("feasible")),
                "solver_status": proposal.get("solver_status"),
                "disagreement_with_baseline": bool(proposal.get("disagreement_with_baseline")),
                "fallback_recommendation": proposal.get("fallback_recommendation"),
            }
        )
        return step

    def finalize(self, *, planner_quality_summary: dict[str, Any]) -> dict[str, Any]:
        write_started = perf_counter()
        _write_jsonl(self.proposal_path, self.proposals)
        proposal_write_ms = float((perf_counter() - write_started) * 1000.0)
        write_started = perf_counter()
        _write_jsonl(self.disagreement_path, self.disagreement_events)
        disagreement_write_ms = float((perf_counter() - write_started) * 1000.0)
        summary = _build_shadow_summary_from_proposals(
            case_id=self.case_id,
            proposals=self.proposals,
            disagreement_events=self.disagreement_events,
            proposal_source="osqp_mpc_shadow_backend_v1",
            quality_proxy_mode="predicted_from_osqp_mpc_shadow_backend",
        )
        realized_outcome_payload = build_shadow_realized_outcome_payload(
            case_id=self.case_id,
            proposals=self.proposals,
            execution_rows=self.timeline_records,
        )
        realized_outcome_summary = dict(realized_outcome_payload.get("summary") or {})
        summary = _merge_realized_outcome_into_shadow_summary(summary, realized_outcome_summary)
        comparison = build_shadow_baseline_comparison_payload(
            case_id=self.case_id,
            shadow_summary=summary,
            planner_quality_summary=planner_quality_summary,
        )
        write_started = perf_counter()
        _write_json(self.summary_path, summary)
        summary_write_ms = float((perf_counter() - write_started) * 1000.0)
        write_started = perf_counter()
        _write_json(self.comparison_path, comparison)
        comparison_write_ms = float((perf_counter() - write_started) * 1000.0)
        write_started = perf_counter()
        _write_jsonl(self.realized_trace_path, list(realized_outcome_payload.get("trace_rows") or []))
        realized_trace_write_ms = float((perf_counter() - write_started) * 1000.0)
        write_started = perf_counter()
        _write_json(self.realized_summary_path, realized_outcome_summary)
        realized_summary_write_ms = float((perf_counter() - write_started) * 1000.0)
        write_started = perf_counter()
        _write_jsonl(self.latency_trace_path, self.latency_trace_rows)
        latency_trace_write_ms = float((perf_counter() - write_started) * 1000.0)
        solver_latencies = [float(row["solver_latency_ms"]) for row in self.latency_trace_rows]
        build_latencies = [float(row.get("problem_build_latency_ms") or 0.0) for row in self.latency_trace_rows]
        setup_latencies = [float(row.get("solver_setup_latency_ms") or 0.0) for row in self.latency_trace_rows]
        compute_latencies = [float(row.get("planner_total_compute_latency_ms") or row["solver_latency_ms"]) for row in self.latency_trace_rows]
        total_latencies = [float(row["shadow_tick_runtime_ms"]) for row in self.latency_trace_rows]
        bookkeeping_latencies = [float(row["bookkeeping_latency_ms"]) for row in self.latency_trace_rows]
        latency_summary = {
            "schema_version": "stage4.shadow_runtime_latency_summary.v1",
            "case_id": self.case_id,
            "num_ticks": len(self.latency_trace_rows),
            "mean_solver_latency_ms": (sum(solver_latencies) / len(solver_latencies)) if solver_latencies else None,
            "mean_problem_build_latency_ms": (sum(build_latencies) / len(build_latencies)) if build_latencies else None,
            "mean_solver_setup_latency_ms": (sum(setup_latencies) / len(setup_latencies)) if setup_latencies else None,
            "mean_planner_total_compute_latency_ms": (sum(compute_latencies) / len(compute_latencies)) if compute_latencies else None,
            "mean_shadow_tick_runtime_ms": (sum(total_latencies) / len(total_latencies)) if total_latencies else None,
            "mean_bookkeeping_latency_ms": (sum(bookkeeping_latencies) / len(bookkeeping_latencies)) if bookkeeping_latencies else None,
            "artifact_write_latency_ms": {
                "proposal_jsonl": proposal_write_ms,
                "disagreement_jsonl": disagreement_write_ms,
                "summary_json": summary_write_ms,
                "comparison_json": comparison_write_ms,
                "realized_trace_jsonl": realized_trace_write_ms,
                "realized_summary_json": realized_summary_write_ms,
                "latency_trace_jsonl": latency_trace_write_ms,
            },
        }
        _write_json(self.latency_summary_path, latency_summary)
        return {
            "proposal_rows": list(self.proposals),
            "disagreement_events": list(self.disagreement_events),
            "latency_trace_rows": list(self.latency_trace_rows),
            "latency_summary": latency_summary,
            "summary": summary,
            "comparison": comparison,
            "realized_outcome_trace": list(realized_outcome_payload.get("trace_rows") or []),
            "realized_outcome_summary": realized_outcome_summary,
        }
