from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.full_authority_ring_stability import run_stage6_full_authority_ring_stability


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 6 full-authority MPC ring stability campaign.")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    payload = run_stage6_full_authority_ring_stability(REPO_ROOT, repeats=int(args.repeats))
    status = str((payload.get("result") or {}).get("overall_status") or "not_ready")
    print(status)
    return 0 if status == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
