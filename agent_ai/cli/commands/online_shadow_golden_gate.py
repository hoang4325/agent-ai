from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.online_shadow_golden import run_stage6_online_shadow_golden_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 6 online shadow golden gate on the promoted two-case subset.")
    parser.add_argument("--repin-baseline", action="store_true")
    args = parser.parse_args()

    result = run_stage6_online_shadow_golden_gate(REPO_ROOT, repin_baseline=bool(args.repin_baseline))
    print(result["gate_result"]["overall_status"])
    return 0 if result["gate_result"]["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
