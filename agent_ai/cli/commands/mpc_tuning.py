from __future__ import annotations

import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.mpc_tuning import run_stage6_mpc_tuning


if __name__ == "__main__":
    result = run_stage6_mpc_tuning(REPO_ROOT)
    overall = ((result.get("decision") or {}).get("final_confirmation") or {}).get("overall_status", "unknown")
    print(overall)
    raise SystemExit(0 if overall == "pass" else 1)
