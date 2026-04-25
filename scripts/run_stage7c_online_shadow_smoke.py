"""
Stage 7C — Agent Shadow Online Smoke Script
=============================================
scripts/run_stage7c_online_shadow_smoke.py

Runs Stage 7C agent shadow online smoke on the exact 2 promoted cases:
  - ml_right_positive_core
  - stop_follow_ambiguity_core

CRITICAL: shadow-only, NO control authority.
  Baseline tactical + MPC remain the sole authority at all times.
  No assist mode. No takeover. No authority hand-off.

Usage:
  python scripts/run_stage7c_online_shadow_smoke.py
  python scripts/run_stage7c_online_shadow_smoke.py --adapter-mode stub
  python scripts/run_stage7c_online_shadow_smoke.py --max-frames 60
    python scripts/run_stage7c_online_shadow_smoke.py --cases ml_right_positive_core,stop_follow_ambiguity_core,ml_left_positive_core,arbitration_stop_during_prepare_right
  python scripts/run_stage7c_online_shadow_smoke.py --output-dir D:/Agent-AI/benchmark/reports/stage7c_run1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.stage7_agent_shadow_online_smoke import run_stage7c_online_shadow_smoke


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 7C Agent Shadow Online Smoke\n"
            "SHADOW ONLY — agent has NO control authority.\n"
            "Baseline tactical + MPC remain sole authority at all times."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-root", default="D:\\Agent-AI", help="Path to Agent-AI workspace root")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional benchmark config override (e.g. a live CARLA report runtime config).",
    )
    parser.add_argument(
        "--set",
        default="stage7_agent_shadow_online_smoke",
        help="Benchmark set name inside the selected config.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Optional comma-separated case override for controlled expansion runs.",
    )
    parser.add_argument(
        "--adapter-mode",
        default="stub",
        choices=["stub", "api", "local"],
        help="Agent adapter mode (stub = deterministic, api = LLM API)",
    )
    parser.add_argument(
        "--adapter-model-id",
        default="stub_v1",
        help="Model identifier for provenance tagging",
    )
    parser.add_argument(
        "--adapter-timeout-s",
        type=float,
        default=1.5,
        help="Agent adapter hard timeout in seconds (default 1.5s for online smoke)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Max frames per case (default 60 for Stage 7C smoke budget)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    case_ids = [case.strip() for case in args.cases.split(",") if case.strip()] if args.cases else None

    print("=" * 65)
    print("STAGE 7C AGENT SHADOW ONLINE SMOKE")
    print("shadow_mode = True  |  agent authority = NONE")
    print("baseline tactical + MPC = SOLE AUTHORITY")
    print("no_control | no_assist | no_takeover")
    print("=" * 65)
    print()

    result = run_stage7c_online_shadow_smoke(
        repo_root=repo_root,
        config_path=args.config,
        set_name=args.set,
        case_ids=case_ids,
        adapter_mode=args.adapter_mode,
        adapter_model_id=args.adapter_model_id,
        adapter_timeout_s=args.adapter_timeout_s,
        max_frames_per_case=args.max_frames,
        output_dir=output_dir,
        allow_synthetic_fallback=False,
    )

    gate = result["gate_result"]
    overall = gate["overall_status"]

    print()
    print("=" * 65)
    print(f"FINAL STATUS: {overall.upper()}")
    print("=" * 65)
    print(f"Passed cases: {gate.get('passed_cases', [])}")
    print(f"Failed cases: {gate.get('failed_cases', [])}")
    print(f"Skipped cases: {gate.get('skipped_cases', [])}")

    if gate.get("hard_failures"):
        print("\nHard failures:")
        for f in gate["hard_failures"]:
            print(f"  ✗ {f}")
    if gate.get("soft_warnings"):
        print("\nSoft warnings:")
        for w in gate["soft_warnings"][:8]:
            print(f"  ⚠ {w}")

    print("\nMetric summary (6 reported metrics):")
    for metric_name, by_case in (gate.get("metric_summary") or {}).items():
        for case_id, v in by_case.items():
            val = v.get("value")
            passed = v.get("threshold_passed")
            status_str = "✓" if passed else ("✗" if passed is False else "?")
            print(f"  {status_str} {case_id} | {metric_name}: {val}")

    readiness = result["readiness_audit"]
    if readiness.get("blockers"):
        print("\nReadiness blockers (online infra):")
        for b in readiness["blockers"]:
            print(f"  ✗ {b}")

    print()
    print("➜ benchmark/stage7_agent_shadow_online_smoke_result.json")
    print("➜ benchmark/stage7_agent_shadow_online_smoke_report.json")
    print()

    if overall not in {"pass", "partial"}:
        sys.exit(1)


if __name__ == "__main__":
    main()
