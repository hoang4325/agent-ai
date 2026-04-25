from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from .evaluation_stage3c_coverage import load_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_SCRIPT = PROJECT_ROOT / "scripts" / "stage3_multilane_scenario.py"


def _run_stage3_multilane_command(arguments: List[str]) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCENARIO_SCRIPT)] + arguments
    return subprocess.run(command, check=True, text=True, capture_output=True)


def probe_left_positive(
    *,
    output_json: str | Path,
    host: str = "127.0.0.1",
    port: int = 2000,
    town: str = "Carla/Maps/Town10HD_Opt",
    top_k: int = 6,
    sample_distance_m: float = 5.0,
    min_forward_length_m: float = 80.0,
) -> List[Dict[str, Any]]:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    _run_stage3_multilane_command(
        [
            "--host",
            str(host),
            "--port",
            str(port),
            "--town",
            str(town),
            "probe",
            "--top-k",
            str(int(top_k)),
            "--sample-distance-m",
            str(float(sample_distance_m)),
            "--min-forward-length-m",
            str(float(min_forward_length_m)),
            "--adjacent-side",
            "left",
            "--output-json",
            str(output_json),
        ]
    )
    payload = load_json(output_json)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected left-positive probe payload: {output_json}")
    return payload


def spawn_left_positive(
    *,
    output_manifest: str | Path,
    probe_json: str | Path,
    rank: int = 0,
    host: str = "127.0.0.1",
    port: int = 2000,
    town: str = "Carla/Maps/Town10HD_Opt",
    blocker_distance_m: float = 24.0,
    adjacent_distance_m: float = 42.0,
    npc_handbrake: bool = True,
) -> Dict[str, Any]:
    probe_candidates = load_json(probe_json)
    if not isinstance(probe_candidates, list) or not probe_candidates:
        raise ValueError(f"No left-positive candidates found in {probe_json}")
    index = max(0, min(int(rank), len(probe_candidates) - 1))
    candidate = dict(probe_candidates[index])

    arguments = [
        "--host",
        str(host),
        "--port",
        str(port),
        "--town",
        str(town),
        "spawn",
        "--adjacent-side",
        "left",
        "--road-id",
        str(int(candidate["road_id"])),
        "--section-id",
        str(int(candidate["section_id"])),
        "--lane-id",
        str(int(candidate["lane_id"])),
        "--s",
        str(float(candidate["s"])),
        "--blocker-distance-m",
        str(float(blocker_distance_m)),
        "--adjacent-distance-m",
        str(float(adjacent_distance_m)),
        "--output-manifest",
        str(output_manifest),
    ]
    if npc_handbrake:
        arguments.append("--npc-handbrake")
    _run_stage3_multilane_command(arguments)
    return load_json(output_manifest)
