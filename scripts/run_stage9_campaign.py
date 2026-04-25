"""
Stage 9 — Campaign Runner (Phase 3)
=====================================
Runs all 12 T9-001→T9-012 scenarios, computes promotion gate metrics,
and saves a JSON campaign report.

Run:
  python scripts/run_stage9_campaign.py [--report-dir benchmark/reports]
                                        [--run-id <label>]

Exit code:
  0 = ALL promotion gates passed
  1 = one or more gates failed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stage9.scenario_runner import ALL_SCENARIOS, ScenarioResult
from stage9.stage9_evaluator import EvaluationReport


# ── Promotion gates ───────────────────────────────────────────────────────────
# A scenario "fails" if any of its checks is False.
# Promotion additionally enforces scenario-level pass rates.

_PROMOTION_TARGETS = {
    "scenario_pass_rate":       0.90,   # ≥ 90% scenarios must pass
    "required_scenarios": [             # These must ALL individually pass
        "T9-001", "T9-005", "T9-007", "T9-008", "T9-009",
    ],
    "phase2_scenarios": [               # Phase 2 gate
        "T9-006", "T9-007", "T9-010", "T9-011",
    ],
}


# ── Report helpers ────────────────────────────────────────────────────────────

def _build_report(
    results: list[ScenarioResult],
    run_id: str,
    total_elapsed_ms: float,
) -> dict:

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    pass_rate = len(passed) / len(results) if results else 0.0

    required_ok = all(
        any(r.scenario_id == sid and r.passed for r in results)
        for sid in _PROMOTION_TARGETS["required_scenarios"]
    )
    phase2_ok = all(
        any(r.scenario_id == sid and r.passed for r in results)
        for sid in _PROMOTION_TARGETS["phase2_scenarios"]
    )
    promotion_pass = (
        pass_rate >= _PROMOTION_TARGETS["scenario_pass_rate"]
        and required_ok
        and phase2_ok
    )

    scenario_details = []
    for r in results:
        checks = [
            {"label": lbl, "passed": ok, "detail": det}
            for lbl, ok, det in r.checks
        ]
        scenario_details.append({
            "id":           r.scenario_id,
            "name":         r.scenario_name,
            "passed":       r.passed,
            "elapsed_ms":   round(r.elapsed_ms, 2),
            "state_trace":  r.state_trace,
            "checks":       checks,
            "error":        r.error,
        })

    return {
        "run_id":             run_id,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "total_scenarios":    len(results),
        "passed_count":       len(passed),
        "failed_count":       len(failed),
        "scenario_pass_rate": round(pass_rate * 100, 1),
        "promotion_gate":     "PASS" if promotion_pass else "FAIL",
        "total_elapsed_ms":   round(total_elapsed_ms, 2),
        "gate_details": {
            "scenario_pass_rate_ok": pass_rate >= _PROMOTION_TARGETS["scenario_pass_rate"],
            "required_scenarios_ok": required_ok,
            "phase2_scenarios_ok":   phase2_ok,
        },
        "scenarios": scenario_details,
        "failed_scenario_ids": [r.scenario_id for r in failed],
    }


def _print_summary(report: dict) -> None:
    SEP = "═" * 70

    print(f"\n{SEP}")
    print(f"  Stage 9 — Campaign Report  |  run_id: {report['run_id']}")
    print(SEP)
    print(f"  Timestamp    : {report['timestamp']}")
    print(f"  Total elapsed: {report['total_elapsed_ms']:.0f} ms")
    print()

    for s in report["scenarios"]:
        ok = "✓" if s["passed"] else "✗"
        print(f"  [{ok}] {s['id']}  {s['name']}")
        if not s["passed"]:
            for chk in s["checks"]:
                tick = "   ✓" if chk["passed"] else "   ✗"
                print(f"  {tick}  {chk['label']}  {chk['detail']}")
        if s.get("error"):
            print(f"       ERROR: {s['error']}")

    print()
    print(f"  Scenarios passed : {report['passed_count']} / {report['total_scenarios']}")
    print(f"  Pass rate        : {report['scenario_pass_rate']}%")
    print()

    gd = report["gate_details"]
    ticks = {True: "✓", False: "✗"}
    print(f"  [{ticks[gd['scenario_pass_rate_ok']]}] Pass rate ≥ 90%")
    print(f"  [{ticks[gd['required_scenarios_ok']]}] Required scenarios "
          f"({', '.join(_PROMOTION_TARGETS['required_scenarios'])}) all pass")
    print(f"  [{ticks[gd['phase2_scenarios_ok']]}] Phase 2 scenarios "
          f"({', '.join(_PROMOTION_TARGETS['phase2_scenarios'])}) all pass")
    print()

    gate = report["promotion_gate"]
    if gate == "PASS":
        print("  ══════════════ PROMOTION GATE: ✓ PASS ══════════════")
        print("  Stage 9 is ready for promotion.")
    else:
        print("  ══════════════ PROMOTION GATE: ✗ FAIL ══════════════")
        print(f"  Failed scenarios: {report['failed_scenario_ids']}")
    print(SEP + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 9 — Campaign Runner")
    parser.add_argument(
        "--report-dir",
        default="benchmark/reports",
        help="Directory to write campaign report and per-scenario logs",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run label (defaults to timestamp)",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("stage9_campaign_%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir)
    log_dir = report_dir / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 70}")
    print(f"  Stage 9 — Campaign  ({len(ALL_SCENARIOS)} scenarios)")
    print(f"  run_id : {run_id}")
    print(f"  logs   : {log_dir}")
    print(f"{'═' * 70}")

    results: list[ScenarioResult] = []
    campaign_start = time.monotonic()

    for scenario_fn in ALL_SCENARIOS:
        sid = scenario_fn.__name__.replace("run_", "").replace("_", "-")
        print(f"\n  ── {sid} ───────────────────────────────────────────────")
        try:
            result = scenario_fn(log_dir)
            results.append(result)
            icon = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  [{icon}] elapsed={result.elapsed_ms:.0f}ms")
            for lbl, ok, detail in result.checks:
                tick = "    ✓" if ok else "    ✗"
                print(f"  {tick} {lbl}{(': ' + detail) if detail else ''}")
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"  [✗ ERROR] {sid}: {exc}")
            results.append(
                ScenarioResult(
                    scenario_id=sid,
                    scenario_name=scenario_fn.__doc__.splitlines()[1].strip() if scenario_fn.__doc__ else sid,
                    passed=False,
                    error=str(exc) + "\n" + tb,
                )
            )

    total_elapsed_ms = (time.monotonic() - campaign_start) * 1000
    report = _build_report(results, run_id, total_elapsed_ms)

    # Save JSON report
    report_path = report_dir / f"{run_id}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _print_summary(report)
    print(f"  Report saved → {report_path}\n")

    return 0 if report["promotion_gate"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
