from __future__ import annotations

import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.mpc_e2e_completion import run_stage6_mpc_e2e_completion


def main() -> int:
    payload = run_stage6_mpc_e2e_completion(REPO_ROOT)
    print(str((payload.get("result") or {}).get("mpc_end_to_end_status") or "not_complete"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
