from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.stage6_mpc_e2e_completion import run_stage6_mpc_e2e_completion
from benchmark.stage6_full_authority_canary import run_stage6_full_authority_canary
from benchmark.stage6_full_authority_ring import run_stage6_full_authority_ring
from benchmark.stage6_full_authority_ring_stability import run_stage6_full_authority_ring_stability
from benchmark.stage6_takeover_rollout_design import run_stage6_takeover_rollout_design
from benchmark.stage6_takeover_rollout_phase import run_stage6_takeover_rollout_phase


def main() -> int:
    run_stage6_takeover_rollout_phase(
        REPO_ROOT,
        phase="phase2",
        operator_signoff_source="interactive_user_request_2026_04_17_phase2",
        kill_switch_verification_source="simulation_abort_path_verified_2026_04_17_phase2",
    )
    run_stage6_takeover_rollout_phase(
        REPO_ROOT,
        phase="phase3",
        operator_signoff_source="interactive_user_request_2026_04_17_phase3",
        kill_switch_verification_source="simulation_abort_path_verified_2026_04_17_phase3",
    )
    run_stage6_takeover_rollout_design(REPO_ROOT)
    run_stage6_full_authority_canary(
        REPO_ROOT,
        phase="phase0",
        operator_signoff_source="interactive_user_request_2026_04_17_full_authority_phase0",
        kill_switch_verification_source="simulation_abort_path_verified_2026_04_17_full_authority_phase0",
    )
    run_stage6_full_authority_canary(
        REPO_ROOT,
        phase="phase1",
        operator_signoff_source="interactive_user_request_2026_04_17_full_authority_phase1",
        kill_switch_verification_source="simulation_abort_path_verified_2026_04_17_full_authority_phase1",
    )
    run_stage6_full_authority_ring(REPO_ROOT)
    run_stage6_full_authority_ring_stability(REPO_ROOT, repeats=3)
    payload = run_stage6_mpc_e2e_completion(REPO_ROOT)
    print(str((payload.get("result") or {}).get("mpc_end_to_end_status") or "not_complete"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
