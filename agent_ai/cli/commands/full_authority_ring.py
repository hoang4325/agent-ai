from __future__ import annotations

import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.takeover.full_authority_ring import run_full_authority_ring


def main() -> int:
    payload = run_full_authority_ring(REPO_ROOT)
    status = str((payload.get("result") or {}).get("overall_status") or "fail")
    print(status)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
