from __future__ import annotations

import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.takeover_enable_gate import run_stage6_takeover_enable_gate


def main() -> int:
    payload = run_stage6_takeover_enable_gate(REPO_ROOT)
    status = str((payload.get("result") or {}).get("takeover_enable_gate_status") or "fail")
    print(status)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
