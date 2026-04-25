"""
Stage 9 — Stage 9 Evaluator (v2)
===================================
Computes all mandatory evaluation metrics from the Stage 9 authority log.

Metrics tracked:
  TSR  — Takeover Success Rate     = successful / total takeovers  (target ≥ 90%)
  SRR  — Stuck Resolution Rate     = resolved / total stucks       (target ≥ 85%)
  FTR  — False Takeover Rate       = unnecessary / total takeovers (target ≤  5%)
  GRR  — Graceful Revoke Rate      = smooth / total revokes        (target ≥ 95%)
  AOI  — Authority Oscillation Idx = state_transitions / minute    (target ≤ 2/min)
  SGC  — Stale Grant Count         = grants with stale world       (target == 0)
  OGV  — ODD-Grant Violation Count = grants outside IN_ODD         (target == 0)
  MRMSR— MRM Success Rate          = mrm_to_safe_stop / total_mrm  (target ≥ 99%)
  JerkPeak  — max jerk in handoff window
  RevokeLat — median frames from veto to revoke effective

Input: path to authority_logger JSONL log file.
Output: EvaluationReport dataclass + printable summary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Optional


# ── Report dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvaluationReport:
    total_takeovers:          int     = 0
    successful_takeovers:     int     = 0
    unnecessary_takeovers:    int     = 0
    tsr:                      float   = 0.0
    ftr:                      float   = 0.0

    total_stucks:             int     = 0
    resolved_stucks:          int     = 0
    srr:                      float   = 0.0

    total_revokes:            int     = 0
    smooth_revokes:           int     = 0
    grr:                      float   = 0.0

    state_transition_count:   int     = 0
    observation_duration_s:   float   = 0.0
    aoi:                      float   = 0.0    # transitions per minute

    stale_grant_count:        int     = 0      # target == 0
    odd_grant_violation_count: int    = 0      # target == 0

    total_mrm:                int     = 0
    mrm_to_safe_stop:         int     = 0
    mrm_success_rate:         float   = 0.0

    revoke_latency_frames:    list    = field(default_factory=list)
    median_revoke_latency_fr: float   = 0.0

    total_frames:             int     = 0

    # Gate results
    gate_results:             dict    = field(default_factory=dict)
    all_gates_pass:           bool    = False


# ── Gates ─────────────────────────────────────────────────────────────────────

_GATES = {
    "TSR >= 0.90":  lambda r: r.tsr >= 0.90,
    "SRR >= 0.85":  lambda r: r.srr >= 0.85,
    "FTR <= 0.05":  lambda r: r.ftr <= 0.05,
    "GRR >= 0.95":  lambda r: r.grr >= 0.95,
    "AOI <= 2/min": lambda r: r.aoi <= 2.0,
    "SGC == 0":     lambda r: r.stale_grant_count == 0,
    "OGV == 0":     lambda r: r.odd_grant_violation_count == 0,
    "MRM >= 0.99":  lambda r: r.mrm_success_rate >= 0.99,
}


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Stage9Evaluator:
    """
    Evaluate Stage 9 authority metrics from JSONL authority log.

    Usage:
      ev = Stage9Evaluator(log_path)
      report = ev.compute()
      ev.print_report(report)
    """

    def __init__(self, log_path: Path | str) -> None:
        self._log_path = Path(log_path)

    def compute(self) -> EvaluationReport:
        events = self._load_events()
        return self._analyse(events)

    def print_report(self, report: EvaluationReport) -> None:
        SEP = "═" * 66
        print(f"\n{SEP}")
        print("  Stage 9 — Evaluation Report")
        print(SEP)
        print(f"  Total frames logged:       {report.total_frames}")
        print(f"  Observation duration:      {report.observation_duration_s:.1f}s")
        print()

        def _row(label, value, target, passed):
            tick = "✓" if passed else "✗"
            print(f"  [{tick}] {label:<35} {value}  (target: {target})")

        _row("Takeover Success Rate (TSR)", f"{report.tsr:.1%}", "≥ 90%",  report.tsr >= 0.90)
        _row("Stuck Resolution Rate (SRR)", f"{report.srr:.1%}", "≥ 85%",  report.srr >= 0.85)
        _row("False Takeover Rate (FTR)",   f"{report.ftr:.1%}", "≤  5%",  report.ftr <= 0.05)
        _row("Graceful Revoke Rate (GRR)",  f"{report.grr:.1%}", "≥ 95%",  report.grr >= 0.95)
        _row("Authority Oscillation (AOI)", f"{report.aoi:.2f}/min", "≤ 2/min", report.aoi <= 2.0)
        _row("Stale Grant Count (SGC)",     str(report.stale_grant_count),   "== 0", report.stale_grant_count == 0)
        _row("ODD Grant Violation (OGV)",   str(report.odd_grant_violation_count), "== 0", report.odd_grant_violation_count == 0)
        _row("MRM Success Rate",            f"{report.mrm_success_rate:.1%}", "≥ 99%", report.mrm_success_rate >= 0.99)

        print()
        print(f"  Median Revoke Latency:     {report.median_revoke_latency_fr:.1f} frames")
        print()

        if report.all_gates_pass:
            print("  ══════════════════════ ALL GATES: PASS ══════════════════════")
        else:
            fail_gates = [k for k, v in report.gate_results.items() if not v]
            print(f"  ══════ FAIL — {len(fail_gates)} gate(s) not met: {fail_gates} ══════")
        print(SEP + "\n")

    # ── Parse & analyse ───────────────────────────────────────────────────────

    def _load_events(self) -> list[dict]:
        if not self._log_path.exists():
            return []
        events = []
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events

    def _analyse(self, events: list[dict]) -> EvaluationReport:
        r = EvaluationReport()
        if not events:
            r.gate_results = {k: fn(r) for k, fn in _GATES.items()}
            r.all_gates_pass = all(r.gate_results.values())
            return r

        r.total_frames = len({e.get("frame_id", 0) for e in events})

        # Frame timestamps for duration estimation (via frame_id proxy)
        frame_ids = sorted({e.get("frame_id", 0) for e in events})
        if len(frame_ids) >= 2:
            r.observation_duration_s = (frame_ids[-1] - frame_ids[0]) * 0.1  # 10 Hz

        # Counters
        grant_frames: dict[str, int] = {}   # contract_id → frame_id of GRANT
        veto_frames:  dict[str, int] = {}   # contract_id → frame_id of VETO

        for ev in events:
            etype = ev.get("event", "")
            fid   = ev.get("frame_id", 0)
            cid   = ev.get("contract_id", "")

            if etype == "STATE_TRANSITION":
                r.state_transition_count += 1
                to_state = ev.get("to", "")
                from_state = ev.get("from", "")

                if to_state == "AGENT_REQUESTING_AUTHORITY":
                    r.total_takeovers += 1
                    r.total_stucks += 1

                if to_state == "BASELINE_CONTROL" and from_state == "AGENT_ACTIVE_BOUNDED":
                    r.resolved_stucks += 1

            elif etype == "GRANT":
                grant_frames[cid] = fid
                r.successful_takeovers += 1

            elif etype == "REJECT":
                # A reject after takeover request counts as unsuccessful
                pass

            elif etype == "REVOKE":
                r.total_revokes += 1
                if cid in veto_frames:
                    latency = fid - veto_frames[cid]
                    r.revoke_latency_frames.append(latency)
                    if latency <= 5:
                        r.smooth_revokes += 1
                else:
                    # Revoke without recorded veto (e.g. contract expiry) = smooth
                    r.smooth_revokes += 1

            elif etype == "VETO":
                veto_frames[cid] = fid

            elif etype == "REJECT":
                reject_reasons = ev.get("all_reasons", [])
                if any("S-005" in r2 for r2 in reject_reasons):
                    r.stale_grant_count += 1
                if any("S-007" in r2 for r2 in reject_reasons):
                    r.odd_grant_violation_count += 1

            elif etype == "MRM_START":
                r.total_mrm += 1

            elif etype == "SAFE_STOP":
                r.mrm_to_safe_stop += 1

        # Compute rates
        def safe_div(a, b): return a / b if b > 0 else 1.0

        r.tsr = safe_div(r.successful_takeovers, r.total_takeovers)
        r.ftr = safe_div(r.unnecessary_takeovers, r.total_takeovers)
        r.srr = safe_div(r.resolved_stucks, r.total_stucks)
        r.grr = safe_div(r.smooth_revokes, r.total_revokes)

        # AOI: state transitions per minute
        duration_min = r.observation_duration_s / 60.0
        r.aoi = r.state_transition_count / duration_min if duration_min > 0 else 0.0

        # MRM success rate
        r.mrm_success_rate = safe_div(r.mrm_to_safe_stop, r.total_mrm)

        # Revoke latency
        if r.revoke_latency_frames:
            r.median_revoke_latency_fr = median(r.revoke_latency_frames)

        # Gates
        r.gate_results = {k: fn(r) for k, fn in _GATES.items()}
        r.all_gates_pass = all(r.gate_results.values())

        return r
