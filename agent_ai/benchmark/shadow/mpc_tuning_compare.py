from __future__ import annotations

from pathlib import Path

from .mpc_tuning import run_mpc_tuning_compare


def main(repo_root: str | Path) -> dict:
    return run_mpc_tuning_compare(repo_root)
