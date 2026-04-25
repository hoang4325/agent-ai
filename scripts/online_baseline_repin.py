from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_decision(
    *,
    stability_report: Path,
    metric_registry_path: Path,
    baseline_dir: Path,
    decision_path: Path,
) -> dict[str, Any]:
    report = _read_json(stability_report) or {}
    registry = _read_json(metric_registry_path) or {}

    stable_cases: list[str] = []
    for case in report.get("cases") or []:
        if case.get("stability_class") in {"stable", "acceptable"}:
            stable_cases.append(str(case.get("case_id")))

    any_hard_candidate = any(
        meta.get("gate_recommendation") == "hard_gate"
        for meta in registry.values()
        if isinstance(meta, dict)
    )

    if len(stable_cases) >= 2 and any_hard_candidate:
        repin_status = "approved"
        reason = "Enough stable online cases and at least one hard-gate metric candidate."
    else:
        repin_status = "deferred"
        reason = "Online reproducibility or metric stability is insufficient for baseline pin."

    payload = {
        "generated_at": utc_now_iso(),
        "selected_cases": stable_cases,
        "repin_status": repin_status,
        "reason": reason,
        "notes": [
            "Require at least two stable/acceptable online cases for first online baseline.",
            "Require hard-gate-ready online metrics with repeatability evidence.",
        ],
    }
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    baseline_dir.mkdir(parents=True, exist_ok=True)
    if repin_status == "approved":
        metrics_payload = {
            "benchmark_version": "system_benchmark_v1_online",
            "generated_at_utc": utc_now_iso(),
            "selected_cases": stable_cases,
            "note": "Initial online baseline pinned.",
        }
        commit_payload = {
            "benchmark_version": "system_benchmark_v1_online",
            "captured_at_utc": utc_now_iso(),
            "source": "online_e2e_smoke_stable_subset",
            "selected_cases": stable_cases,
        }
        (baseline_dir / "online_baseline_metrics.json").write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
        (baseline_dir / "online_baseline_commit.json").write_text(json.dumps(commit_payload, indent=2) + "\n", encoding="utf-8")

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide whether to pin initial online baseline.")
    parser.add_argument("--stability-report", default="benchmark/reports/online_stability/online_smoke_stability_report.json")
    parser.add_argument("--metric-stability-registry", default="benchmark/reports/online_stability/online_metric_stability_registry.json")
    parser.add_argument("--baseline-dir", default="benchmark/baselines/system_benchmark_v1_online")
    parser.add_argument("--decision-output", default="benchmark/reports/online_stability/online_baseline_decision.json")
    args = parser.parse_args()

    payload = run_decision(
        stability_report=Path(args.stability_report),
        metric_registry_path=Path(args.metric_stability_registry),
        baseline_dir=Path(args.baseline_dir),
        decision_path=Path(args.decision_output),
    )
    print(json.dumps({"repin_status": payload.get("repin_status"), "selected_cases": payload.get("selected_cases")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
