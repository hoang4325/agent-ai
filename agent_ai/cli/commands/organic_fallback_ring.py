from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
PROJECT_ROOT = REPO_ROOT
from agent_ai.benchmark.takeover.organic_fallback_ring import run_organic_fallback_ring  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 6 organic fallback evidence ring.")
    parser.add_argument("--case-id", dest="case_ids", action="append", default=None)
    parser.add_argument("--carla-root", default="D:/carla")
    parser.add_argument("--retry-attempts", type=int, default=1)
    args = parser.parse_args()
    result = run_organic_fallback_ring(
        PROJECT_ROOT,
        case_ids=args.case_ids,
        carla_root=args.carla_root,
        retry_attempts=args.retry_attempts,
    )
    print(json.dumps(result.get("result") or {}, indent=2))


if __name__ == "__main__":
    main()
