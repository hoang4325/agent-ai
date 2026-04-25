"""
run_stage5a_gate.py
====================
Entrypoint to run the full Stage 5A gate pipeline:
1. Contract audit
2. Metric pack generation
3. Failure replay case generation
4. Benchmark execution on stage5a_core_golden + stage5a_failure_replay
5. Gate result with final readiness verdict
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.frozen_corpus_builder import build_frozen_corpus
from benchmark.stage5a_contract_audit import run_contract_audit
from benchmark.stage5a_metric_pack import build_stage5a_metric_pack
from benchmark.stage5a_failure_replay_builder import build_failure_replay_cases
from benchmark.io import dump_json, load_yaml, resolve_repo_path
from scripts.run_system_benchmark_e2e import run_e2e_benchmark


def _merge_gate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple benchmark gate results into a single Stage 5A gate verdict."""
    all_failed = []
    all_warnings = []
    all_notes = []

    for result in results:
        gate = result.get("gate_result", {})
        all_failed.extend(gate.get("failed_cases", []))
        all_warnings.extend(gate.get("warning_cases", []))
        all_notes.extend(gate.get("notes", []))
        diagnostics = result.get("scenario_replay_diagnostics", {})
        all_failed.extend(diagnostics.get("skipped_cases", []))
        all_failed.extend(diagnostics.get("runtime_failed_cases", []))
        all_notes.extend(diagnostics.get("notes", []))

    overall = "pass"
    if all_failed:
        overall = "fail"
    elif all_warnings:
        overall = "partial"

    return {
        "overall_status": overall,
        "failed_cases": sorted(set(all_failed)),
        "warning_cases": sorted(set(all_warnings)),
        "notes": all_notes,
    }


def _load_stage5a_case_sets(repo_root: Path) -> dict[str, list[str]]:
    benchmark_cfg = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    result: dict[str, list[str]] = {}
    for set_name in ["stage5a_core_golden", "stage5a_failure_replay"]:
        set_path = resolve_repo_path(repo_root, benchmark_cfg["sets"][set_name])
        payload = load_yaml(set_path)
        result[set_name] = list(payload.get("cases", []))
    return result


def _scenario_replay_diagnostics(result: dict[str, Any], required_cases: list[str]) -> dict[str, Any]:
    rows = list(result.get("mode_resolution") or [])
    row_by_case = {str(row.get("scenario_id")): row for row in rows}
    skipped_cases = []
    runtime_failed_cases = []
    successful_cases = []
    notes: list[str] = []

    for case_id in required_cases:
        row = row_by_case.get(case_id)
        if not row:
            skipped_cases.append(case_id)
            notes.append(f"scenario_replay_missing_mode_resolution:{case_id}")
            continue
        status = str(row.get("status"))
        if status == "success":
            successful_cases.append(case_id)
        elif status == "skipped":
            skipped_cases.append(case_id)
            notes.append(f"scenario_replay_skipped:{case_id}:{row.get('reason')}")
        elif status == "failed":
            runtime_failed_cases.append(case_id)
            notes.append(f"scenario_replay_runtime_failed:{case_id}:{row.get('reason')}")

    return {
        "required_cases": required_cases,
        "successful_cases": successful_cases,
        "skipped_cases": skipped_cases,
        "runtime_failed_cases": runtime_failed_cases,
        "notes": notes,
    }


def run_stage5a_gate(
    repo_root: str | Path | None = None,
    skip_audit: bool = False,
    skip_failure_cases: bool = False,
) -> dict[str, Any]:
    """Execute the complete Stage 5A gate pipeline."""
    repo_root = Path(repo_root or REPO_ROOT)

    print("=" * 72)
    print("  STAGE 5A GATE PIPELINE")
    print("=" * 72)

    # ── Phase 1: Contract Audit ───────────────────────────────────────────
    audit_result = None
    if not skip_audit:
        print("\n[Phase 1] Running contract audit...")
        try:
            audit_result = run_contract_audit(repo_root)
            print(f"  → Audited {audit_result['summary']['total_cases_audited']} cases across {audit_result['summary']['total_contracts']} contract layers")
            print(f"  → Found {audit_result['summary']['total_ambiguous_fields']} ambiguous fields")
        except Exception as exc:
            print(f"  ✗ Contract audit failed: {exc}")

    # ── Phase 2: Metric Pack ─────────────────────────────────────────────
    print("\n[Phase 2] Building metric pack...")
    try:
        metric_pack = build_stage5a_metric_pack(repo_root)
        print(f"  → {len(metric_pack['hard_gate_metrics'])} hard gate metrics")
        print(f"  → {len(metric_pack['soft_guardrail_metrics'])} soft guardrail metrics")
        print(f"  → {len(metric_pack['diagnostic_metrics'])} diagnostic metrics")
    except Exception as exc:
        print(f"  ✗ Metric pack build failed: {exc}")
        metric_pack = None

    # ── Phase 3: Failure Replay Cases ─────────────────────────────────────
    if not skip_failure_cases:
        print("\n[Phase 3] Generating failure replay cases...")
        try:
            created = build_failure_replay_cases(repo_root)
            print(f"  → Created {len(created)} failure replay case files")
        except Exception as exc:
            print(f"  ✗ Failure replay case generation failed: {exc}")

    benchmark_cfg = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    frozen_root = resolve_repo_path(
        repo_root,
        ((benchmark_cfg.get("frozen_corpus") or {}).get("root")) or "benchmark/frozen_corpus/v1",
    )
    if frozen_root is None:
        raise ValueError("frozen corpus root is missing from benchmark config")
    stage5a_sets = _load_stage5a_case_sets(repo_root)

    # ── Phase 4: Refresh Frozen Corpus ────────────────────────────────────
    print("\n[Phase 4] Refreshing frozen corpus...")
    frozen_build = build_frozen_corpus(
        repo_root=repo_root,
        config_path="benchmark/benchmark_v1.yaml",
        output_root=frozen_root,
    )
    print(f"  → Frozen corpus refreshed at {frozen_root}")
    print(f"  → Readiness counts: {frozen_build['readiness']['summary']['readiness_counts']}")

    # ── Phase 5: Run Stage 5A Core Golden via scenario_replay ────────────
    print("\n[Phase 5] Running stage5a_core_golden via scenario_replay...")
    benchmark_results = []
    try:
        golden_result = run_e2e_benchmark(
            config_path=repo_root / "benchmark" / "benchmark_v1.yaml",
            set_name="stage5a_core_golden",
            benchmark_mode="scenario_replay",
            execute_stages=True,
        )
        golden_result["scenario_replay_diagnostics"] = _scenario_replay_diagnostics(
            golden_result,
            stage5a_sets["stage5a_core_golden"],
        )
        benchmark_results.append(golden_result)
        gate = golden_result.get("gate_result", {})
        print(f"  → Golden status: {gate['overall_status']}")
        if gate.get("failed_cases"):
            print(f"  → Failed: {gate['failed_cases']}")
        diagnostics = golden_result["scenario_replay_diagnostics"]
        if diagnostics["skipped_cases"]:
            print(f"  → Skipped in scenario_replay: {diagnostics['skipped_cases']}")
        if diagnostics["runtime_failed_cases"]:
            print(f"  → Runtime failed: {diagnostics['runtime_failed_cases']}")
    except Exception as exc:
        print(f"  ✗ Golden benchmark failed: {exc}")

    # ── Phase 6: Run Stage 5A Failure Replay via scenario_replay ─────────
    print("\n[Phase 6] Running stage5a_failure_replay via scenario_replay...")
    try:
        replay_result = run_e2e_benchmark(
            config_path=repo_root / "benchmark" / "benchmark_v1.yaml",
            set_name="stage5a_failure_replay",
            benchmark_mode="scenario_replay",
            execute_stages=True,
        )
        replay_result["scenario_replay_diagnostics"] = _scenario_replay_diagnostics(
            replay_result,
            stage5a_sets["stage5a_failure_replay"],
        )
        benchmark_results.append(replay_result)
        gate = replay_result.get("gate_result", {})
        print(f"  → Failure replay status: {gate['overall_status']}")
        if gate.get("failed_cases"):
            print(f"  → Failed: {gate['failed_cases']}")
        diagnostics = replay_result["scenario_replay_diagnostics"]
        if diagnostics["skipped_cases"]:
            print(f"  → Skipped in scenario_replay: {diagnostics['skipped_cases']}")
        if diagnostics["runtime_failed_cases"]:
            print(f"  → Runtime failed: {diagnostics['runtime_failed_cases']}")
    except Exception as exc:
        print(f"  ✗ Failure replay benchmark failed: {exc}")

    # ── Phase 7: Final Gate Verdict ───────────────────────────────────────
    print("\n[Phase 7] Computing final Stage 5A gate verdict...")
    merged_gate = _merge_gate_results(benchmark_results)

    gate_output = {
        "schema_version": "stage5a_gate_result_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage": "5A",
        "benchmark_mode": "scenario_replay",
        "gate_driver": "frozen_corpus + scenario_replay",
        "overall_status": merged_gate["overall_status"],
        "failed_contracts": merged_gate["failed_cases"],
        "failed_cases": merged_gate["failed_cases"],
        "warnings": merged_gate["warning_cases"],
        "notes": merged_gate["notes"],
        "threshold_authoritative_source": {
            "primary": "case_spec.metric_thresholds",
            "fallback": "benchmark_v1.default_metric_thresholds",
            "runner_resolution": "benchmark.runner_core._case_metric_thresholds",
        },
        "audit_summary": (audit_result.get("summary") if audit_result else None),
        "metric_pack_summary": {
            "hard_gate_count": len(metric_pack["hard_gate_metrics"]) if metric_pack else 0,
            "soft_guardrail_count": len(metric_pack["soft_guardrail_metrics"]) if metric_pack else 0,
        },
        "frozen_corpus_summary": frozen_build["readiness"]["summary"],
        "benchmark_runs": [
            {
                "set": r.get("set"),
                "requested_mode": r.get("requested_mode"),
                "status": r.get("gate_result", {}).get("overall_status"),
                "failed_cases": r.get("gate_result", {}).get("failed_cases", []),
                "report_dir": r.get("report_dir"),
                "scenario_replay_diagnostics": r.get("scenario_replay_diagnostics"),
            }
            for r in benchmark_results
        ],
    }

    output_path = repo_root / "benchmark" / "stage5a_gate_result.json"
    dump_json(output_path, gate_output)

    # ── Print verdict ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  STAGE 5A GATE VERDICT: {merged_gate['overall_status'].upper()}")
    print("=" * 72)
    if merged_gate["failed_cases"]:
        print(f"  Failed cases: {merged_gate['failed_cases']}")
    if merged_gate["warning_cases"]:
        print(f"  Warnings: {merged_gate['warning_cases']}")
    print(f"\n  Gate result written to: {output_path}")

    return gate_output


if __name__ == "__main__":
    from typing import Any
    result = run_stage5a_gate()
    sys.exit(0 if result["overall_status"] == "pass" else 1)
