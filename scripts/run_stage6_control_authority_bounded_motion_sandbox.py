from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.stage6_control_authority_handoff_sandbox import (  # noqa: E402
    run_stage6_control_authority_bounded_motion_sandbox,
)


def main() -> None:
    result = run_stage6_control_authority_bounded_motion_sandbox(PROJECT_ROOT)
    print(json.dumps(result.get("result") or {}, indent=2))


if __name__ == "__main__":
    main()
