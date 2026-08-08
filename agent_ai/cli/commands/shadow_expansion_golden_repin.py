from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.shadow.shadow_expansion import rerun_stage6_shadow_expansion_golden_from_stability


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repin the Stage 6 six-case shadow golden baseline from the ready stability envelope and rerun golden."
    )
    parser.add_argument("--carla-root", default="D:/carla")
    parser.add_argument("--carla-port", type=int, default=2000)
    args = parser.parse_args()

    result = rerun_stage6_shadow_expansion_golden_from_stability(
        REPO_ROOT,
        carla_root=args.carla_root,
        carla_port=int(args.carla_port),
    )
    overall = str(((result.get("golden") or {}).get("gate_result") or {}).get("overall_status", "fail"))
    print(overall)
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
