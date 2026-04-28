from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paper-ready blocked-clear figures from Stage10 A/B summaries."
    )
    parser.add_argument(
        "--summary-json",
        action="append",
        required=True,
        help="Path to a tables_ab_summary.json file. Pass once for each blocked-clear case.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where PNG figures will be written.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Stage10 Blocked-Lane",
        help="Optional title prefix for the rendered figures.",
    )
    return parser.parse_args()


def _base_case_name(case_name: str) -> str:
    if case_name.endswith("_assist"):
        return case_name[: -len("_assist")]
    if case_name.endswith("_baseline"):
        return case_name[: -len("_baseline")]
    return case_name


def _display_case_name(base_case: str) -> str:
    mapping = {
        "blocked_lane_clear_right": "Blocked-Clear Right",
        "blocked_lane_clear_left": "Blocked-Clear Left",
    }
    return mapping.get(base_case, base_case.replace("_", " ").title())


def _load_rows(summary_paths: List[Path]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for path in summary_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        table2_rows = payload.get("table2_rows") or []
        table3_rows = payload.get("table3_rows") or []
        table3_by_case = {str(row.get("case")): row for row in table3_rows}
        for row in table2_rows:
            case = str(row.get("case"))
            base_case = _base_case_name(case)
            variant = "assist" if case.endswith("_assist") else "baseline" if case.endswith("_baseline") else "other"
            grouped.setdefault(base_case, {})
            merged = dict(row)
            merged.update(table3_by_case.get(case) or {})
            grouped[base_case][variant] = merged
    return grouped


def _require_variant(grouped: Dict[str, Dict[str, Dict[str, Any]]], variant: str) -> None:
    missing = [base_case for base_case, variants in grouped.items() if variant not in variants]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Missing '{variant}' rows for cases: {joined}")


def _plot_route_progress(grouped: Dict[str, Dict[str, Dict[str, Any]]], output_dir: Path, title_prefix: str) -> Path:
    base_cases = sorted(grouped.keys())
    labels = [_display_case_name(base_case) for base_case in base_cases]
    baseline_vals = [
        float(grouped[base_case]["baseline"].get("route_progress_m") or 0.0)
        for base_case in base_cases
    ]
    assist_vals = [
        float(grouped[base_case]["assist"].get("route_progress_m") or 0.0)
        for base_case in base_cases
    ]

    x = np.arange(len(base_cases))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=180)
    baseline_bars = ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#9aa5b1")
    assist_bars = ax.bar(x + width / 2, assist_vals, width, label="Baseline + Agent Assist", color="#2f6fed")

    ax.set_ylabel("Route Progress (m)")
    ax.set_title(f"{title_prefix}: Forward Progress")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)

    ymax = max(baseline_vals + assist_vals + [1.0])
    ax.set_ylim(0, ymax * 1.22)

    for bars in (baseline_bars, assist_bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    path = output_dir / "blocked_clear_route_progress.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_agent_intervention(grouped: Dict[str, Dict[str, Dict[str, Any]]], output_dir: Path, title_prefix: str) -> Path:
    base_cases = sorted(grouped.keys())
    labels = [_display_case_name(base_case) for base_case in base_cases]
    queried_vals = [
        float(grouped[base_case]["assist"].get("assist_agent_query_frames") or 0.0)
        for base_case in base_cases
    ]
    applied_vals = [
        float(grouped[base_case]["assist"].get("assist_applied_frames") or 0.0)
        for base_case in base_cases
    ]

    x = np.arange(len(base_cases))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=180)
    queried_bars = ax.bar(x - width / 2, queried_vals, width, label="Agent Queried Frames", color="#7bc8a4")
    applied_bars = ax.bar(x + width / 2, applied_vals, width, label="Assist Applied Frames", color="#f59e0b")

    ax.set_ylabel("Frames")
    ax.set_title(f"{title_prefix}: Agent Intervention")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)

    ymax = max(queried_vals + applied_vals + [1.0])
    ax.set_ylim(0, ymax * 1.20)

    for bars in (queried_bars, applied_bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    path = output_dir / "blocked_clear_agent_intervention.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    args = _parse_args()
    summary_paths = [Path(p) for p in args.summary_json]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = _load_rows(summary_paths)
    if not grouped:
        raise SystemExit("No rows found in the provided summary JSON files.")
    _require_variant(grouped, "assist")
    _require_variant(grouped, "baseline")

    route_plot = _plot_route_progress(grouped, output_dir, str(args.title_prefix))
    intervention_plot = _plot_agent_intervention(grouped, output_dir, str(args.title_prefix))

    print(f"Saved: {route_plot}")
    print(f"Saved: {intervention_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
