from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.takeover.full_authority_canary import run_full_authority_canary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 6 full-authority MPC canary.")
    parser.add_argument("--phase", choices=("phase0", "phase1"), default="phase0")
    parser.add_argument("--operator-signoff-source", default="interactive_user_request")
    parser.add_argument("--kill-switch-verification-source", default="simulation_abort_path_verified")
    args = parser.parse_args()

    payload = run_full_authority_canary(
        REPO_ROOT,
        phase=str(args.phase),
        operator_signoff_source=str(args.operator_signoff_source),
        kill_switch_verification_source=str(args.kill_switch_verification_source),
    )
    status = str((payload.get("result") or {}).get("overall_status") or "fail")
    print(status)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
