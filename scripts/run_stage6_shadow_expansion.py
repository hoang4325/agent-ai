from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.stage6_shadow_expansion import run_stage6_shadow_expansion


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 6 six-case shadow expansion.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--carla-root", default="D:/carla")
    parser.add_argument("--carla-port", type=int, default=2000)
    args = parser.parse_args()

    result = run_stage6_shadow_expansion(
        REPO_ROOT,
        repeats=int(args.repeats),
        carla_root=args.carla_root,
        carla_port=int(args.carla_port),
        restart_between_runs=True,
    )
    overall = (result.get("stability") or {}).get("readiness", "not_ready")
    print(overall)
    return 0 if overall == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
