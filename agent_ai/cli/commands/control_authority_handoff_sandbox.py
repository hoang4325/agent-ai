from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.takeover.control_authority_handoff_sandbox import run_control_authority_handoff_sandbox


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 6 no-motion control-authority handoff sandbox.")
    parser.add_argument("--carla-root", default="D:/carla")
    parser.add_argument("--carla-port", type=int, default=2000)
    args = parser.parse_args()

    result = run_control_authority_handoff_sandbox(
        REPO_ROOT,
        carla_root=args.carla_root,
        carla_port=int(args.carla_port),
    )
    overall = str((result.get("result") or {}).get("overall_status") or "not_ready")
    print(overall)
    return 0 if overall == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
