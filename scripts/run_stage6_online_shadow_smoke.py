from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.stage6_online_shadow_smoke import run_stage6_online_shadow_smoke


if __name__ == "__main__":
    result = run_stage6_online_shadow_smoke(REPO_ROOT)
    print(result["gate_result"]["overall_status"])
    raise SystemExit(0 if result["gate_result"]["overall_status"] == "pass" else 1)
