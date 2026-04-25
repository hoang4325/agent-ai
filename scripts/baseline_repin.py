from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.io import dump_json, load_json, load_yaml, resolve_repo_path  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_decision_path() -> Path:
    return REPO_ROOT / "benchmark" / "reports" / "baseline_repin_decision.json"


def _golden_report_dirs() -> list[Path]:
    reports_dir = REPO_ROOT / "benchmark" / "reports"
    return sorted([p for p in reports_dir.glob("golden_*") if p.is_dir()])


def _candidate_from_dir(report_dir: Path) -> dict[str, Any]:
    gate = load_json(report_dir / "gate_result.json", {}) or {}
    benchmark_report = load_json(report_dir / "benchmark_report.json", {}) or {}
    overall_status = gate.get("overall_status")
    warnings = list(gate.get("warning_cases") or [])
    failed = list(gate.get("failed_cases") or [])
    semantic_degraded = list(gate.get("semantic_degraded_cases") or [])
    semantic_skipped = list(gate.get("semantic_skipped_cases") or [])
    benchmark_cases = benchmark_report.get("cases") or []
    case_soft_warnings = [c.get("scenario_id") for c in benchmark_cases if c.get("soft_warnings")]
    approved = (
        overall_status == "pass"
        and not failed
        and not warnings
        and not semantic_degraded
        and not case_soft_warnings
    )
    return {
        "report_dir": str(report_dir),
        "overall_status": overall_status,
        "failed_cases": failed,
        "warning_cases": warnings,
        "semantic_degraded_cases": semantic_degraded,
        "semantic_skipped_cases": semantic_skipped,
        "case_soft_warnings": case_soft_warnings,
        "approved_candidate": approved,
    }


def _select_candidate(candidates: list[dict[str, Any]], selected_run: str | None) -> dict[str, Any] | None:
    if selected_run:
        for c in candidates:
            if Path(c["report_dir"]).name == selected_run:
                return c
        return None
    approved = [c for c in candidates if c.get("approved_candidate")]
    if not approved:
        return None
    approved.sort(key=lambda c: c["report_dir"])
    return approved[-1]


def _apply_repin(selected: dict[str, Any], semantic_hardening_version: str) -> dict[str, Any]:
    report_dir = Path(selected["report_dir"])
    baseline_dir = REPO_ROOT / "benchmark" / "baselines" / "system_benchmark_v1"
    baseline_metrics = baseline_dir / "baseline_metrics.json"
    baseline_commit = baseline_dir / "baseline_commit.json"

    previous_commit = load_json(baseline_commit, {}) or {}
    previous_source = previous_commit.get("source_report_dir")

    snapshot_path = report_dir / "baseline_snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"baseline snapshot missing: {snapshot_path}")

    shutil.copy2(snapshot_path, baseline_metrics)

    commit_payload = {
        "benchmark_version": "system_benchmark_v1",
        "captured_at_utc": _utc_now(),
        "workspace_root": str(REPO_ROOT),
        "git_commit": None,
        "workspace_is_git_repo": False,
        "pinned_from_set": "golden",
        "source_report_dir": str(report_dir),
        "source_gate_result": str(report_dir / "gate_result.json"),
        "semantic_hardening_version": semantic_hardening_version,
        "previous_baseline_source_report_dir": previous_source,
        "note": "Repinned after v1.1 semantic/binding cleanup with stable golden pass and no warnings.",
    }
    dump_json(baseline_commit, commit_payload)
    return {
        "baseline_metrics": str(baseline_metrics),
        "baseline_commit": str(baseline_commit),
        "previous_baseline_source_report_dir": previous_source,
    }


def run_repin(config_path: Path, decision_output: Path, selected_run: str | None, apply: bool) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    candidates = [_candidate_from_dir(p) for p in _golden_report_dirs()]
    selected = _select_candidate(candidates, selected_run)

    approved = selected is not None and bool(selected.get("approved_candidate"))
    repin_status = "approved" if approved else "deferred"
    reason = "stable golden pass candidate selected" if approved else "no fully clean golden candidate"

    apply_result = None
    if apply and approved and selected is not None:
        apply_result = _apply_repin(selected, semantic_hardening_version="v1.1")

    payload = {
        "benchmark_version": cfg.get("version"),
        "generated_at": _utc_now(),
        "candidate_golden_runs": candidates,
        "selected_run": (Path(selected["report_dir"]).name if selected else None),
        "repin_status": repin_status,
        "reason": reason,
        "remaining_warnings": ([] if approved else ["golden_not_clean"]),
        "apply_requested": apply,
        "apply_result": apply_result,
    }
    dump_json(decision_output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide and optionally repin benchmark baseline from golden runs.")
    parser.add_argument("--config", default=str(REPO_ROOT / "benchmark" / "benchmark_v1.yaml"))
    parser.add_argument("--decision-output", default=str(_default_decision_path()))
    parser.add_argument("--selected-run", default=None, help="Optional explicit golden run directory name, e.g. golden_20260412_123456")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    payload = run_repin(Path(args.config), Path(args.decision_output), args.selected_run, args.apply)
    print(json.dumps({
        "selected_run": payload["selected_run"],
        "repin_status": payload["repin_status"],
        "decision_output": args.decision_output,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
