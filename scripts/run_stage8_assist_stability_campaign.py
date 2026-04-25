"""
Stage 8 — Assist Stability Campaign
=====================================
Runs the Stage 8 Assist Canary Sandbox multiple times (5-10 runs)
to formally verify stability, AE approval rates, and contract safety
before promoting to Stage 9 Takeover Design.

Usage:
  python scripts/run_stage8_assist_stability_campaign.py --runs 5 --adapter-mode api
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_stage8_assist_smoke import run_stage8_assist_smoke
from benchmark.io import dump_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=5, help="Number of independent runs")
    p.add_argument("--adapter-mode", default="api_simulated", choices=["api_simulated", "api"])
    p.add_argument("--api-key-env", default="AGENT_API_KEY")
    p.add_argument("--api-endpoint", default="https://api.groq.com/openai/v1/chat/completions")
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--repo-root", default=str(REPO_ROOT))
    args = p.parse_args()

    print(f"\n[Stage8-Campaign] ═══════════════════════════════════════════════════")
    print(f"[Stage8-Campaign] Starting Stability Campaign: {args.runs} RUNS")
    print(f"[Stage8-Campaign] Case: ml_right_positive_core | Mode: {args.adapter_mode}")
    print(f"[Stage8-Campaign] ═══════════════════════════════════════════════════\n")

    campaign_fails = 0
    total_approved = 0
    total_vetoed = 0
    total_contract_breaches = 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_dir = Path(args.repo_root) / "benchmark" / "reports" / f"stage8_campaign_{ts}"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = []

    for i in range(1, args.runs + 1):
        print(f"\n[Stage8-Campaign] ─── RUN {i} / {args.runs} ─────────────────────────────")
        
        try:
            res = run_stage8_assist_smoke(
                repo_root=args.repo_root,
                adapter_mode=args.adapter_mode,
                api_key_env_var=args.api_key_env,
                api_endpoint=args.api_endpoint,
                max_frames=args.max_frames,
            )
            
            gate = res["gate_result"]
            run_dirs.append(res["run_dir"])
            
            if gate["overall_status"] != "pass":
                campaign_fails += 1
            
            mets = gate.get("metrics", {})
            total_approved += mets.get("ae_approved", 0)
            total_vetoed += mets.get("ae_vetoed", 0)
            total_contract_breaches += mets.get("brp_contract_breach_count", 0)

        except Exception as e:
            print(f"[Stage8-Campaign] Run {i} FAILED WITH EXCEPTION: {e}")
            campaign_fails += 1

    total_brps = max(total_approved + total_vetoed, 1)
    mean_approval = total_approved / total_brps

    overall_status = "PASS" if campaign_fails == 0 and mean_approval >= 0.80 and total_contract_breaches == 0 else "FAIL"

    report = {
        "campaign_id": f"stage8_campaign_{ts}",
        "runs": args.runs,
        "adapter_mode": args.adapter_mode,
        "overall_status": overall_status,
        "runs_failed": campaign_fails,
        "aggregate_metrics": {
            "total_approved": total_approved,
            "total_vetoed": total_vetoed,
            "mean_approval_rate": round(mean_approval, 4),
            "total_contract_breaches": total_contract_breaches,
        },
        "run_directories": run_dirs
    }

    dump_json(campaign_dir / "campaign_report.json", report)
    dump_json(Path(args.repo_root) / "benchmark" / "stage8_assist_campaign_report.json", report)

    print(f"\n[Stage8-Campaign] ══════════════════════ CAMPAIGN: {overall_status} ══════════════════════")
    print(f"[Stage8-Campaign] Runs Passed: {args.runs - campaign_fails} / {args.runs}")
    print(f"[Stage8-Campaign] Aggregate Approval Rate: {mean_approval:.2%} (Target >= 80%)")
    print(f"[Stage8-Campaign] Total Contract Breaches: {total_contract_breaches} (Target == 0)")
    print(f"[Stage8-Campaign] Report saved to: benchmark/stage8_assist_campaign_report.json\n")


if __name__ == "__main__":
    main()
