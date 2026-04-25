from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.stage6_organic_fallback_ring import run_stage6_organic_fallback_ring  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 6 organic fallback evidence ring.")
    parser.add_argument("--case-id", dest="case_ids", action="append", default=None)
    parser.add_argument("--carla-root", default="D:/carla")
    parser.add_argument("--retry-attempts", type=int, default=1)
    args = parser.parse_args()
    result = run_stage6_organic_fallback_ring(
        PROJECT_ROOT,
        case_ids=args.case_ids,
        carla_root=args.carla_root,
        retry_attempts=args.retry_attempts,
    )
    print(json.dumps(result.get("result") or {}, indent=2))


if __name__ == "__main__":
    main()
