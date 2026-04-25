"""
Stage 7 — LLM/API Adapter Shadow Smoke
========================================

Opens LLM/API adapter smoke for Stage 7 shadow on exactly 2 cases:
  - ml_right_positive_core
  - stop_follow_ambiguity_core

CRITICAL INVARIANTS (same as stub smoke):
  - shadow_mode = True always
  - Agent has NO control / trajectory / MPC authority
  - Baseline tactical + MPC remain sole authority at all times
  - No assist mode, no takeover, no authority hand-off
  - timeout / malformed / invalid → fallback_to_baseline recorded cleanly

Adapter modes:
  api_simulated  [DEFAULT] — intelligent local LLM shim; no API key required.
                             Full pipeline smoke: parse, validate, contract,
                             artifact materialize, disagreement events.
                             Disagrees with baseline on ~22% of frames.

  api            [PRODUCTION] — real Gemini call via urllib.
                             Requires AGENT_API_KEY env var.
                             Missing key → all frames timeout-fallback (recorded).

Usage:
  python scripts/run_stage7_llm_shadow_smoke.py
  python scripts/run_stage7_llm_shadow_smoke.py --adapter-mode api_simulated
  python scripts/run_stage7_llm_shadow_smoke.py --adapter-mode api
  python scripts/run_stage7_llm_shadow_smoke.py --adapter-mode api --api-key-env GEMINI_API_KEY

Output:
  benchmark/stage7_llm_shadow_smoke_result.json
  benchmark/stage7_llm_shadow_smoke_report.json
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.stage7_llm_shadow_smoke import run_stage7_llm_shadow_smoke


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 7 LLM/API adapter shadow smoke (2-case, shadow-only)"
    )
    p.add_argument(
        "--adapter-mode",
        default="api_simulated",
        choices=["api_simulated", "api"],
        help="Adapter mode: api_simulated (no key needed) or api (requires AGENT_API_KEY)",
    )
    p.add_argument(
        "--api-key-env",
        default="AGENT_API_KEY",
        help="Env var name for real API key (only used with --adapter-mode api)",
    )
    p.add_argument(
        "--api-endpoint",
        default=None,
        help="Override Gemini API endpoint URL (optional)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Max frames per case (default 60)",
    )
    p.add_argument(
        "--disagree-rate",
        type=float,
        default=0.22,
        help="api_simulated: fraction of frames with disagreement (default 0.22)",
    )
    p.add_argument(
        "--latency-ms",
        type=float,
        default=35.0,
        help="api_simulated: simulated per-frame latency in ms (default 35.0)",
    )
    p.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repo root path",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.adapter_mode == "api":
        key = os.environ.get(args.api_key_env, "").strip()
        if not key:
            print(
                f"\n[Stage7-LLM] WARNING: --adapter-mode api selected but "
                f"{args.api_key_env} is not set.\n"
                f"All frames will timeout-fallback to baseline (valid contract behavior).\n"
                f"Set {args.api_key_env}=<your-key> for real LLM calls.\n"
                f"Use --adapter-mode api_simulated to run without a key.\n"
            )
        else:
            print(f"[Stage7-LLM] API key found in {args.api_key_env} — real Gemini calls enabled.")

    result = run_stage7_llm_shadow_smoke(
        repo_root=args.repo_root,
        adapter_mode=args.adapter_mode,
        api_key_env_var=args.api_key_env,
        api_endpoint=args.api_endpoint,
        max_frames_per_case=args.max_frames,
        api_simulated_disagree_rate=args.disagree_rate,
        api_simulated_latency_ms=args.latency_ms,
    )

    gate = result["gate_result"]
    status = gate["overall_status"].upper()
    print(f"\n[Stage7-LLM] ══════════════════════ FINAL: {status} ══════════════════════")
    print(f"[Stage7-LLM] Result  → benchmark/stage7_llm_shadow_smoke_result.json")
    print(f"[Stage7-LLM] Report  → benchmark/stage7_llm_shadow_smoke_report.json")

    if gate.get("hard_failures"):
        print("[Stage7-LLM] Hard failures:")
        for hf in gate["hard_failures"]:
            print(f"  ✗ {hf}")
    if gate.get("soft_warnings"):
        print("[Stage7-LLM] Soft warnings:")
        for sw in gate["soft_warnings"]:
            print(f"  ⚠ {sw}")

    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
