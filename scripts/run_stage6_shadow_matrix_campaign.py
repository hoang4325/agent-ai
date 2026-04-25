from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.stage6_shadow_matrix_campaign import run_stage6_shadow_matrix_campaign


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run repeated Stage 6 six-case shadow matrix campaign."
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--carla-root", default="D:/carla")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--retry-attempts", type=int, default=1)
    args = parser.parse_args()

    result = run_stage6_shadow_matrix_campaign(
        REPO_ROOT,
        repeats=int(args.repeats),
        carla_root=args.carla_root,
        carla_port=int(args.carla_port),
        retry_attempts=int(args.retry_attempts),
    )
    overall = str((result.get("result") or {}).get("overall_status") or "fail")
    print(overall)
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

