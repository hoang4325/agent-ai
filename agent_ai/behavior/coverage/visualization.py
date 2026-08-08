from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt


def save_stage3c_coverage_visualization(
    *,
    output_dir: str | Path,
    coverage_summary: Dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gap_analysis = coverage_summary["gap_analysis"]
    execution_summary = coverage_summary["execution_coverage_summary"]

    scenario_labels = ["right_positive", "left_positive", "junction_execution"]
    pass_values = [1.0 if gap_analysis[label]["status"] == "pass" else 0.0 for label in scenario_labels]
    lane_change_successes = [
        0 if execution_summary[label] is None else int(execution_summary[label].get("lane_change_successes", 0))
        for label in scenario_labels
    ]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    axes[0].bar(scenario_labels, pass_values, color=["tab:green" if value > 0 else "tab:red" for value in pass_values])
    axes[0].set_ylim(0.0, 1.1)
    axes[0].set_ylabel("Pass status")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(scenario_labels, lane_change_successes, color="tab:blue")
    axes[1].set_ylabel("Lane-change successes")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "execution_coverage_overview.png", dpi=160)
    plt.close(fig)
