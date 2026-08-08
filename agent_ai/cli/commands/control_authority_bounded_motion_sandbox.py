from __future__ import annotations

import json
from pathlib import Path
import sys

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
PROJECT_ROOT = REPO_ROOT
from agent_ai.benchmark.control_authority_handoff_sandbox import (  # noqa: E402
    run_stage6_control_authority_bounded_motion_sandbox,
)


def main() -> None:
    result = run_stage6_control_authority_bounded_motion_sandbox(PROJECT_ROOT)
    print(json.dumps(result.get("result") or {}, indent=2))


if __name__ == "__main__":
    main()
