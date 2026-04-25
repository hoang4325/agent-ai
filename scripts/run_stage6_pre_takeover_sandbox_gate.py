from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.stage6_pre_takeover_sandbox import run_stage6_pre_takeover_sandbox_gate


def main() -> int:
    result = run_stage6_pre_takeover_sandbox_gate(REPO_ROOT)
    sandbox_gate_status = str((result.get("result") or {}).get("sandbox_gate_status") or "fail")
    print(sandbox_gate_status)
    return 0 if sandbox_gate_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
