from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.stage6_mpc_tuning import run_stage6_mpc_tuning


if __name__ == "__main__":
    result = run_stage6_mpc_tuning(REPO_ROOT)
    overall = ((result.get("decision") or {}).get("final_confirmation") or {}).get("overall_status", "unknown")
    print(overall)
    raise SystemExit(0 if overall == "pass" else 1)
