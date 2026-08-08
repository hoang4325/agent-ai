from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.takeover.takeover_ring_6case_stability import run_takeover_ring_6case_stability


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 6 six-case takeover ring stability campaign.")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    payload = run_takeover_ring_6case_stability(REPO_ROOT, repeats=int(args.repeats))
    status = str(payload.get("readiness") or "not_ready")
    print(status)
    return 0 if status == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
