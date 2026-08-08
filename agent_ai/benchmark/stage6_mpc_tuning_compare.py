from __future__ import annotations

from pathlib import Path

from .stage6_mpc_tuning import run_stage6_mpc_tuning_compare


def main(repo_root: str | Path) -> dict:
    return run_stage6_mpc_tuning_compare(repo_root)
