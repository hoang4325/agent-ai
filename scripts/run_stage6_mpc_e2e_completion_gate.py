from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.stage6_mpc_e2e_completion import run_stage6_mpc_e2e_completion


def main() -> int:
    payload = run_stage6_mpc_e2e_completion(REPO_ROOT)
    print(str((payload.get("result") or {}).get("mpc_end_to_end_status") or "not_complete"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
