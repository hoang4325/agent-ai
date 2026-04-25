from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.stage6_takeover_ring_6case_sandbox import run_stage6_takeover_ring_6case_sandbox


def main() -> int:
    payload = run_stage6_takeover_ring_6case_sandbox(REPO_ROOT)
    status = str((payload.get("result") or {}).get("overall_status") or "fail")
    print(status)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
