from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.stage6_takeover_rollout_design import run_stage6_takeover_rollout_design


def main() -> int:
    payload = run_stage6_takeover_rollout_design(REPO_ROOT)
    status = str((payload.get("result") or {}).get("rollout_design_status") or "not_ready")
    print(status)
    return 0 if status == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
