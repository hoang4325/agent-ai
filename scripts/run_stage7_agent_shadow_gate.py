"""
Stage 7 — Agent Shadow Gate (PHASE A7.6)
==========================================
run_stage7_agent_shadow_gate.py

Entrypoint for the Stage 7 agent shadow gate.
Runs agent shadow over frozen corpus replay or scenario replay artifacts,
materializes all Stage 7 shadow artifacts, compares vs baseline, validates
contract, and produces the gate result.

Usage:
  python scripts/run_stage7_agent_shadow_gate.py \\
    --repo-root D:\\Agent-AI \\
    --set stage7_agent_shadow_replay \\
    [--adapter-mode stub|api] \\
    [--adapter-model-id stub_v1] \\
    [--output-dir benchmark/reports/stage7_shadow_YYYYMMDD]

CRITICAL: This gate is SHADOW ONLY.
  - Baseline tactical + MPC are the sole authority at all times.
  - Agent shadow output is for analysis only.
  - No control fields are emitted or respected.
  - If agent times out / invalid → fallback logged, baseline used for agent record.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.agent_shadow_adapter import AgentShadowAdapter, AgentShadowAdapterConfig, build_adapter
from benchmark.io import dump_json, dump_jsonl, load_json, load_jsonl, load_yaml, resolve_repo_path
from benchmark.stage7_agent_shadow_audit import run_stage7_contract_audit
from benchmark.stage7_agent_shadow_metric_pack import (
    build_agent_baseline_comparison,
    build_agent_shadow_summary,
    compute_agent_shadow_metrics,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_set_cases(set_name: str, benchmark_config: dict[str, Any], repo_root: Path) -> list[str]:
    sets = benchmark_config.get("sets") or {}
    set_path_str = sets.get(set_name)
    if not set_path_str:
        raise ValueError(f"Set '{set_name}' not in benchmark_v1.yaml sets. Add it or use an existing set.")
    set_path = resolve_repo_path(repo_root, set_path_str)
    payload = load_yaml(set_path)
    return list(payload.get("cases", []))


def _load_case_artifacts(case_id: str, repo_root: Path, cases_dir: Path) -> dict[str, Any]:
    """Load case spec and its relevant baseline artifacts."""
    case_path = cases_dir / f"{case_id}.yaml"
    if not case_path.exists():
        return {}
    spec = load_yaml(case_path)
    artifacts = spec.get("artifacts") or {}
    result = {"spec": spec, "artifacts": artifacts}

    # Load baseline behavior timeline from Stage 3B
    bt_path = resolve_repo_path(repo_root, artifacts.get("stage3b_behavior_timeline"))
    result["baseline_timeline"] = load_jsonl(bt_path) if bt_path else []

    # Load world state sample
    ws_path = resolve_repo_path(repo_root, artifacts.get("stage2_world_state_sample"))
    result["world_state_sample"] = load_json(ws_path) if ws_path else None

    # Load lane context sample
    lane_path = resolve_repo_path(repo_root, artifacts.get("stage3a_lane_world_state_sample"))
    result["lane_world_state_sample"] = load_json(lane_path) if lane_path else None

    # Load behavior request sample
    br_path = resolve_repo_path(repo_root, artifacts.get("stage3b_behavior_request_sample"))
    result["behavior_request_sample"] = load_json(br_path) if br_path else None

    # Load stop target context
    stop_path = resolve_repo_path(repo_root, artifacts.get("stage3c_stop_target"))
    result["stop_target"] = load_json(stop_path) if stop_path else None

    return result


def _build_agent_input_bundle(
    frame_record: dict[str, Any],
    world_state_sample: dict[str, Any] | None,
    lane_world_state_sample: dict[str, Any] | None,
    stop_target: dict[str, Any] | None,
) -> tuple[dict, list, dict, dict, dict, dict]:
    """Extract the 6-tuple input bundle for the agent adapter from frame artifacts."""
    # Ego state from world state (sample-level)
    ws = world_state_sample or {}
    ego_state = dict(ws.get("ego") or {})
    ego_state["risk_summary"] = dict(ws.get("risk_summary") or {})
    ego_state["scene"] = dict(ws.get("scene") or {})

    # Tracked objects
    tracked_objects = list(ws.get("objects") or [])

    # Lane context from Stage 3A
    lane_ws = dict(lane_world_state_sample or {})
    lane_context = dict(lane_ws.get("lane_context") or {})
    lane_context["lane_scene"] = dict(lane_ws.get("lane_scene") or {})
    lane_context["lane_relative_objects"] = list(lane_ws.get("lane_relative_objects") or [])

    # Route context from Stage 3B behavior_request_v2
    br2 = dict((frame_record.get("behavior_request_v2") or {}))
    route_context = {
        "route_option": br2.get("route_option"),
        "preferred_lane": br2.get("preferred_lane"),
        "route_mode": br2.get("route_mode"),
        "route_conflict_flags": list(br2.get("route_conflict_flags") or []),
        "distance_to_next_branch_m": br2.get("longitudinal_horizon_m"),
    }

    # Stop context
    stop_ctx: dict[str, Any] = {}
    if stop_target:
        stop_ctx = {
            "binding_status": stop_target.get("binding_status"),
            "distance_to_stop_m": stop_target.get("distance_to_stop_m"),
        }

    # Baseline context
    baseline_context = {
        "requested_behavior": br2.get("requested_behavior"),
        "target_lane": br2.get("target_lane"),
        "lane_change_permission": dict(br2.get("lane_change_permission") or {}),
        "confidence": br2.get("confidence"),
        "stop_reason": br2.get("stop_reason"),
    }

    return ego_state, tracked_objects, lane_context, route_context, stop_ctx, baseline_context


def run_stage7_gate(
    *,
    repo_root: Path,
    benchmark_config: dict[str, Any],
    set_name: str,
    output_dir: Path,
    adapter: AgentShadowAdapter,
) -> dict[str, Any]:
    """
    Core gate runner:
      1. Load cases from set
      2. For each case, iterate baseline timeline frames
      3. Call adapter per-frame in shadow mode
      4. Produce all 4 required artifacts per-case
      5. Compute metrics
      6. Build gate result
    """
    cases_dir = resolve_repo_path(repo_root, benchmark_config["cases_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = _load_set_cases(set_name, benchmark_config, repo_root)
    if not case_ids:
        raise ValueError(f"No cases found in set '{set_name}'")

    print(f"[Stage7] Starting gate '{set_name}' with {len(case_ids)} case(s) — shadow_only mode")

    all_case_results: list[dict[str, Any]] = []
    gate_passed = True
    gate_hard_failures: list[str] = []
    gate_soft_warnings: list[str] = []

    for case_id in case_ids:
        print(f"[Stage7]   Case: {case_id}")
        case_data = _load_case_artifacts(case_id, repo_root, cases_dir)
        if not case_data:
            print(f"[Stage7]     WARN: case spec not found — skipping {case_id}")
            all_case_results.append({
                "case_id": case_id,
                "status": "skipped",
                "reason": "case_spec_not_found",
            })
            continue

        baseline_timeline = case_data.get("baseline_timeline") or []
        world_state_sample = case_data.get("world_state_sample")
        lane_world_state_sample = case_data.get("lane_world_state_sample")
        stop_target = case_data.get("stop_target")

        if not baseline_timeline:
            print(f"[Stage7]     WARN: no baseline timeline — skipping {case_id}")
            all_case_results.append({
                "case_id": case_id,
                "status": "skipped",
                "reason": "baseline_timeline_empty",
            })
            continue

        # Per-case output directory
        case_out = output_dir / case_id / "stage7"
        case_out.mkdir(parents=True, exist_ok=True)

        # Run agent shadow over all frames
        intent_records: list[dict[str, Any]] = []
        disagreement_events: list[dict[str, Any]] = []

        for frame_idx, frame_record in enumerate(baseline_timeline):
            frame_id = int(frame_record.get("frame_id", frame_idx))

            ego, tracked, lane_ctx, route_ctx, stop_ctx, baseline_ctx = _build_agent_input_bundle(
                frame_record=frame_record,
                world_state_sample=world_state_sample,
                lane_world_state_sample=lane_world_state_sample,
                stop_target=stop_target,
            )

            intent = adapter.call(
                case_id=case_id,
                frame_id=frame_id,
                ego_state=ego,
                tracked_objects=tracked,
                lane_context=lane_ctx,
                route_context=route_ctx,
                stop_context=stop_ctx,
                baseline_context=baseline_ctx,
            )
            intent_dict = asdict(intent)
            intent_records.append(intent_dict)

            # Collect disagreement events
            if not intent_dict.get("agrees_with_baseline", True):
                disagreement_events.append({
                    "case_id": case_id,
                    "frame_id": frame_id,
                    "baseline_intent": intent_dict.get("baseline_intent"),
                    "agent_intent": intent_dict.get("tactical_intent"),
                    "agent_confidence": intent_dict.get("confidence"),
                    "disagreement_useful": intent_dict.get("disagreement_useful"),
                    "reason_tags": intent_dict.get("reason_tags"),
                    "fallback_to_baseline": intent_dict.get("fallback_to_baseline"),
                    "validation_status": intent_dict.get("validation_status"),
                    "route_context_summary": intent_dict.get("route_context_summary"),
                })

        # Write per-case artifacts
        comparison_path = case_out / "agent_baseline_comparison.json"
        disagreement_path = case_out / "agent_disagreement_events.jsonl"
        intent_path = case_out / "agent_shadow_intent.jsonl"
        summary_path = case_out / "agent_shadow_summary.json"

        dump_jsonl(intent_path, intent_records)
        dump_jsonl(disagreement_path, disagreement_events)

        summary = build_agent_shadow_summary(case_id=case_id, intents=intent_records)
        comparison = build_agent_baseline_comparison(case_id=case_id, intents=intent_records)
        dump_json(summary_path, summary)
        dump_json(comparison_path, comparison)

        # Compute metrics
        metrics = compute_agent_shadow_metrics(
            intents=intent_records,
            summary=summary,
            comparison_path=comparison_path,
            disagreement_path=disagreement_path,
        )

        # Evaluate gate
        case_hard_failures: list[str] = []
        case_soft_warnings: list[str] = []
        for metric_name, metric_result in metrics.items():
            passed = metric_result.get("threshold_passed")
            reason = metric_result.get("threshold_reason", "")
            gate_usage = metric_result.get("gate_usage", "diagnostic_only")
            if passed is False:
                if gate_usage == "hard_gate":
                    case_hard_failures.append(f"{metric_name}: {reason}")
                elif gate_usage == "soft_guardrail":
                    case_soft_warnings.append(f"{metric_name}: {reason}")

        case_status = "pass" if not case_hard_failures else "fail"
        if case_hard_failures:
            gate_passed = False
            gate_hard_failures.extend([f"{case_id}/{f}" for f in case_hard_failures])
        if case_soft_warnings:
            gate_soft_warnings.extend([f"{case_id}/{f}" for f in case_soft_warnings])

        print(f"[Stage7]     Frames: {len(intent_records)} | "
              f"Valid: {summary.get('valid_frames', 0)} | "
              f"Fallback: {summary.get('fallback_frames', 0)} | "
              f"Disagree: {summary.get('disagreement_frames', 0)} | "
              f"Status: {case_status.upper()}")

        all_case_results.append({
            "case_id": case_id,
            "status": case_status,
            "hard_failures": case_hard_failures,
            "soft_warnings": case_soft_warnings,
            "metrics": {k: {"value": v.get("value"), "threshold_passed": v.get("threshold_passed")} for k, v in metrics.items()},
            "summary": summary,
            "artifacts": {
                "agent_shadow_intent": str(intent_path),
                "agent_shadow_summary": str(summary_path),
                "agent_baseline_comparison": str(comparison_path),
                "agent_disagreement_events": str(disagreement_path),
            },
        })

    # ── Write session-level outputs ──────────────────────────────────────────
    overall_status = "pass" if gate_passed else "fail"
    print(f"\n[Stage7] Overall gate: {overall_status.upper()}")

    # Derive stage label from set name (7A / 7B / 7C)
    _STAGE_MAP = {
        "stage7_agent_shadow_replay": "7A",
        "stage7_agent_shadow_scenario_replay": "7B",
        "stage7_agent_shadow_online_smoke": "7C",
    }
    stage_label = _STAGE_MAP.get(set_name, "7")

    gate_result = {
        "schema_version": "stage7_agent_shadow_gate_result_v1",
        "generated_at_utc": _utc_now(),
        "set": set_name,
        "stage": stage_label,
        "mode": "shadow_only",
        "agent_authority": "shadow_only — baseline tactical + MPC remain sole authority",
        "overall_status": overall_status,
        "hard_failures": gate_hard_failures,
        "soft_warnings": gate_soft_warnings,
        "case_count": len(case_ids),
        "passed_cases": [r["case_id"] for r in all_case_results if r.get("status") == "pass"],
        "failed_cases": [r["case_id"] for r in all_case_results if r.get("status") == "fail"],
        "skipped_cases": [r["case_id"] for r in all_case_results if r.get("status") == "skipped"],
    }

    report = {
        "schema_version": "stage7_agent_shadow_report_v1",
        "generated_at_utc": _utc_now(),
        "set": set_name,
        "stage": stage_label,
        "mode": "shadow_only",
        "agent_authority": "shadow_only",
        "overall_status": overall_status,
        "cases": all_case_results,
    }

    # Named output files (set-specific for 7C)
    gate_filename = f"stage7_agent_shadow_online_smoke_result.json" if stage_label == "7C" else "stage7_agent_shadow_gate_result.json"
    report_filename = f"stage7_agent_shadow_online_smoke_report.json" if stage_label == "7C" else "stage7_agent_shadow_report.json"

    dump_json(output_dir / gate_filename, gate_result)
    dump_json(output_dir / report_filename, report)

    print(f"[Stage7] Gate result  → {output_dir / gate_filename}")
    print(f"[Stage7] Full report  → {output_dir / report_filename}")

    return {
        "gate_result": gate_result,
        "report": report,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 7 Agent Shadow Gate — SHADOW ONLY, baseline authority unchanged",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-root", default="D:\\Agent-AI", help="Path to Agent-AI workspace root")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional benchmark config override. Defaults to benchmark/benchmark_v1.yaml.",
    )
    parser.add_argument(
        "--set",
        default="stage7_agent_shadow_replay",
        choices=[
            "stage7_agent_shadow_replay",
            "stage7_agent_shadow_scenario_replay",
            "stage7_agent_shadow_online_smoke",
        ],
        help="Benchmark set to run",
    )
    parser.add_argument(
        "--adapter-mode",
        default="stub",
        choices=["stub", "api", "local"],
        help="Agent adapter mode (stub = deterministic benchmark stub, api = LLM API)",
    )
    parser.add_argument("--adapter-model-id", default="stub_v1", help="Model identifier for provenance tagging")
    parser.add_argument("--adapter-timeout-s", type=float, default=2.0, help="Agent adapter hard timeout in seconds")
    parser.add_argument("--output-dir", default=None, help="Override output directory (default: benchmark/reports/stage7_...)")
    parser.add_argument("--run-audit", action="store_true", help="Run Stage 7 contract audit before gate execution")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    benchmark_config_path = Path(args.config).resolve() if args.config else (repo_root / "benchmark" / "benchmark_v1.yaml")
    benchmark_config = load_yaml(benchmark_config_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        repo_root / "benchmark" / "reports" / f"stage7_{args.set}_{ts}"
    )

    # Optional: run audit first
    if args.run_audit:
        print("[Stage7] Running contract audit...")
        run_stage7_contract_audit(repo_root)

    # Build adapter
    adapter = build_adapter(
        mode=args.adapter_mode,
        model_id=args.adapter_model_id,
        api_timeout_s=args.adapter_timeout_s,
    )

    print(f"[Stage7] Adapter: mode={args.adapter_mode}, model_id={args.adapter_model_id}")
    print(f"[Stage7] Set: {args.set}")
    print(f"[Stage7] Output: {output_dir}")
    print("[Stage7] INVARIANT: agent is shadow-only. Baseline tactical + MPC = sole authority.\n")

    result = run_stage7_gate(
        repo_root=repo_root,
        benchmark_config=benchmark_config,
        set_name=args.set,
        output_dir=output_dir,
        adapter=adapter,
    )

    gate_status = result["gate_result"]["overall_status"]
    print(f"\n[Stage7] === GATE RESULT: {gate_status.upper()} ===")
    if gate_status != "pass":
        print("[Stage7] Hard failures:")
        for failure in result["gate_result"].get("hard_failures", []):
            print(f"         - {failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()
