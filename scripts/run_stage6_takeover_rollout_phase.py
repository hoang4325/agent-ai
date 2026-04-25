from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.stage6_takeover_rollout_phase import run_stage6_takeover_rollout_phase


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 6 explicit takeover rollout phase automation.")
    parser.add_argument("--phase", choices=("phase2", "phase3"), required=True)
    parser.add_argument("--operator-signoff-source", default="interactive_user_request")
    parser.add_argument("--kill-switch-verification-source", default="simulation_abort_path_verified")
    args = parser.parse_args()

    payload = run_stage6_takeover_rollout_phase(
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
