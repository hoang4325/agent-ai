from __future__ import annotations

import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.shadow.online_shadow_smoke import run_online_shadow_smoke


if __name__ == "__main__":
    result = run_online_shadow_smoke(REPO_ROOT)
    print(result["gate_result"]["overall_status"])
    raise SystemExit(0 if result["gate_result"]["overall_status"] == "pass" else 1)
