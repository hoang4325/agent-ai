"""
Stage 5B Set Builder
====================
Synchronizes the Stage 5B core/failure set YAML files from the metric pack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .stage5b_metric_pack import STAGE5B_CASE_REQUIREMENTS


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def build_stage5b_sets(repo_root: str | Path) -> dict[str, str]:
    repo_root = Path(repo_root)
    sets_dir = repo_root / "benchmark" / "sets"
    written: dict[str, str] = {}
    for set_name in ["stage5b_core_golden", "stage5b_failure_replay"]:
        requirement = STAGE5B_CASE_REQUIREMENTS[set_name]
        payload = {
            "set_name": set_name,
            "description": requirement["description"],
            "cases": list(requirement["cases"]),
        }
        output_path = sets_dir / f"{set_name}.yaml"
        _write_yaml(output_path, payload)
        written[set_name] = str(output_path)
        print(f"[Stage5B] Set file written to {output_path}")
    return written
