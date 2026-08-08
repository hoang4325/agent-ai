from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.online_shadow_stability import run_stage6_online_shadow_stability


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeated Stage 6 online shadow smoke passes and analyze stability.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--carla-root", default="D:/carla")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--no-restart-between-runs", action="store_true")
    args = parser.parse_args()

    result = run_stage6_online_shadow_stability(
        REPO_ROOT,
        repeats=args.repeats,
        carla_root=args.carla_root,
        carla_port=args.carla_port,
        restart_between_runs=not args.no_restart_between_runs,
    )
    print(result["readiness"])
    return 0 if result["readiness"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
