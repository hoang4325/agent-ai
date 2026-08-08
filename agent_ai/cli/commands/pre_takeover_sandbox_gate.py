from __future__ import annotations

import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.takeover.pre_takeover_sandbox import run_pre_takeover_sandbox_gate


def main() -> int:
    result = run_pre_takeover_sandbox_gate(REPO_ROOT)
    sandbox_gate_status = str((result.get("result") or {}).get("sandbox_gate_status") or "fail")
    print(sandbox_gate_status)
    return 0 if sandbox_gate_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
