from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.takeover.non_timeout_fallback_ring import run_non_timeout_fallback_ring


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 6 non-timeout fallback evidence ring.")
    parser.add_argument("--carla-root", default="D:/carla")
    parser.add_argument("--carla-port", type=int, default=2000)
    args = parser.parse_args()

    result = run_non_timeout_fallback_ring(
        REPO_ROOT,
        carla_root=args.carla_root,
        carla_port=int(args.carla_port),
    )
    overall = str((result.get("result") or {}).get("overall_status") or "not_ready")
    print(overall)
    return 0 if overall == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
