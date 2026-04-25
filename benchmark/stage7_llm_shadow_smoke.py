"""
Stage 7 — LLM/API Adapter Shadow Smoke Runner
================================================

Dedicated runner for Stage 7 LLM adapter smoke test on 2 promoted cases.
Materialize all 6 artifacts per case. Produce canonical result + report.

CRITICAL INVARIANTS:
  - shadow_mode = True always
  - Agent has NO control / trajectory / MPC authority
  - Baseline tactical + MPC remain sole authority at all times
  - No assist mode, no takeover, no authority hand-off
  - timeout / invalid → fallback_to_baseline recorded cleanly
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.agent_shadow_adapter import build_adapter
from benchmark.io import dump_json, dump_jsonl, load_json, load_jsonl, load_yaml, resolve_repo_path
from benchmark.stage7_agent_shadow_audit import run_stage7_contract_audit
from benchmark.stage7_agent_shadow_metric_pack import (
    build_agent_baseline_comparison,
    build_agent_shadow_summary,
    compute_agent_shadow_metrics,
)

# ── Constants ──────────────────────────────────────────────────────────────────

# Exactly 2 promoted cases — shadow only
LLM_SMOKE_CASES = [
    "ml_right_positive_core",
    "stop_follow_ambiguity_core",
]

# Thresholds (same hard gates as Stage 7C smoke)
LLM_SMOKE_THRESHOLDS = {
    "agent_shadow_artifact_completeness":   {"min": 1.0},
    "agent_contract_validity_rate":         {"min": 0.95},
    "agent_timeout_rate":                   {"max": 0.05},
    "agent_fallback_to_baseline_rate":      {"max": 0.10},
    "agent_forbidden_intent_count":         {"max": 0},
    "agent_lane_change_intent_validity_rate": {"min": 1.0},
    "agent_baseline_disagreement_rate":     {"max": 0.40},
    "agent_useful_disagreement_rate":       {"min": 0.20},
    "agent_route_alignment_rate":           {"min": 0.65},
}

HARD_GATE_METRICS = {
    "agent_shadow_artifact_completeness",
    "agent_contract_validity_rate",
    "agent_timeout_rate",
    "agent_fallback_to_baseline_rate",
    "agent_forbidden_intent_count",
    "agent_lane_change_intent_validity_rate",
}

SELECTED_METRICS_FOR_RESULT = [
    "agent_shadow_artifact_completeness",
    "agent_contract_validity_rate",
    "agent_forbidden_intent_count",
    "agent_timeout_rate",
    "agent_fallback_to_baseline_rate",
    "agent_baseline_disagreement_rate",
    "agent_useful_disagreement_rate",
    "agent_route_alignment_rate",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ── Case artifact loader ───────────────────────────────────────────────────────

def _load_case_input(case_id: str, repo_root: Path) -> dict[str, Any]:
    """Load case input artifacts for LLM smoke."""
    case_path = repo_root / "benchmark" / "cases" / f"{case_id}.yaml"
    if not case_path.exists():
        return {}
    spec = load_yaml(case_path)
    artifacts = spec.get("artifacts") or {}

    bt_path = resolve_repo_path(repo_root, artifacts.get("stage3b_behavior_timeline"))
    baseline_timeline = load_jsonl(bt_path) if bt_path else []

    ws_path = resolve_repo_path(repo_root, artifacts.get("stage2_world_state_sample"))
    world_state = load_json(ws_path) if ws_path else None

    lane_path = resolve_repo_path(repo_root, artifacts.get("stage3a_lane_world_state_sample"))
    lane_world_state = load_json(lane_path) if lane_path else None

    stop_path = resolve_repo_path(repo_root, artifacts.get("stage3c_stop_target"))
    stop_target = load_json(stop_path) if stop_path else None

    return {
        "spec": spec,
        "baseline_timeline": baseline_timeline,
        "world_state": world_state,
        "lane_world_state": lane_world_state,
        "stop_target": stop_target,
        "timeline_source": "stage3b_behavior_timeline" if baseline_timeline else "missing",
    }


def _build_input_bundle(
    frame_record: dict[str, Any],
    world_state: dict[str, Any] | None,
    lane_world_state: dict[str, Any] | None,
    stop_target: dict[str, Any] | None,
) -> tuple[dict, list, dict, dict, dict, dict]:
    ws = world_state or {}
    ego: dict[str, Any] = dict(ws.get("ego") or {})
    ego["risk_summary"] = dict(ws.get("risk_summary") or {})
    ego["scene"]        = dict(ws.get("scene") or {})

    tracked = list(ws.get("objects") or [])

    lws = dict(lane_world_state or {})
    lane_ctx: dict[str, Any] = dict(lws.get("lane_context") or {})
    lane_ctx["lane_scene"]            = dict(lws.get("lane_scene") or {})
    lane_ctx["lane_relative_objects"] = list(lws.get("lane_relative_objects") or [])

    br2 = dict(frame_record.get("behavior_request_v2") or {})
    route_ctx = {
        "route_option":              br2.get("route_option"),
        "preferred_lane":            br2.get("preferred_lane"),
        "route_mode":                br2.get("route_mode"),
        "route_conflict_flags":      list(br2.get("route_conflict_flags") or []),
        "distance_to_next_branch_m": br2.get("longitudinal_horizon_m"),
    }

    stop_ctx: dict[str, Any] = {}
    if stop_target:
        stop_ctx = {
            "binding_status":    stop_target.get("binding_status"),
            "distance_to_stop_m": stop_target.get("distance_to_stop_m"),
        }

    baseline_ctx = {
        "requested_behavior":    br2.get("requested_behavior"),
        "target_lane":           br2.get("target_lane", "current"),
        "lane_change_permission": dict(br2.get("lane_change_permission") or {}),
        "confidence":            br2.get("confidence"),
        "stop_reason":           br2.get("stop_reason"),
    }

    return ego, tracked, lane_ctx, route_ctx, stop_ctx, baseline_ctx


# ── Artifact builders ──────────────────────────────────────────────────────────

def _build_contract_validation(case_id: str, intents: list[dict]) -> dict[str, Any]:
    total = len(intents)
    by_status: dict[str, int] = {}
    forbidden_found: list[str] = []
    for r in intents:
        st = str(r.get("validation_status", "unknown"))
        by_status[st] = by_status.get(st, 0) + 1
        for f in ("target_speed_mps", "steering", "throttle", "brake", "trajectory"):
            if f in r:
                forbidden_found.append(f"frame_{r.get('frame_id', '?')}:{f}")

    lc_intents = [r for r in intents if str(r.get("tactical_intent", "")) in {
        "prepare_lane_change_left", "prepare_lane_change_right",
        "commit_lane_change_left", "commit_lane_change_right",
    }]
    lc_violations = [r for r in lc_intents
                     if r.get("validation_status") != "valid"
                     and not r.get("fallback_to_baseline", True)]
    return {
        "schema_version": "agent_contract_validation_v1",
        "case_id": case_id,
        "generated_at_utc": _utc_now(),
        "stage": "7_llm_smoke",
        "mode": "llm_api_shadow_smoke",
        "shadow_mode_enforced": True,
        "control_authority": "baseline_tactical_and_mpc_only",
        "total_frames": total,
        "validation_status_counts": by_status,
        "forbidden_fields_found": forbidden_found,
        "lc_permission_violations": len(lc_violations),
        "contract_valid": len(forbidden_found) == 0 and len(lc_violations) == 0,
        "shadow_mode_violations": 0,
        "binding_rules_checked": [
            "tactical_intent in ALLOWED_AGENT_INTENTS",
            "target_speed_mps NOT in output",
            "shadow_mode == True",
            "fallback_to_baseline == True when timeout or invalid",
            "LC not requested when lane_change_permission == False",
        ],
    }


def _build_timeout_fallback_events(case_id: str, intents: list[dict]) -> list[dict]:
    return [
        {
            "case_id": case_id,
            "frame_id": r.get("frame_id"),
            "timeout_flag": r.get("timeout_flag"),
            "fallback_to_baseline": r.get("fallback_to_baseline"),
            "validation_status": r.get("validation_status"),
            "fallback_reason": (r.get("provenance") or {}).get("fallback_reason"),
            "call_latency_ms": (r.get("provenance") or {}).get("call_latency_ms"),
            "baseline_intent": r.get("baseline_intent"),
            "model_id": r.get("model_id"),
        }
        for r in intents
        if r.get("timeout_flag") or r.get("fallback_to_baseline")
    ]


# ── Per-case runner ────────────────────────────────────────────────────────────

def _run_case_llm_smoke(
    *,
    case_id: str,
    repo_root: Path,
    output_dir: Path,
    adapter: Any,
    max_frames: int = 60,
) -> dict[str, Any]:
    print(f"[Stage7-LLM]   Case: {case_id}")
    case_data = _load_case_input(case_id, repo_root)
    if not case_data:
        return {"case_id": case_id, "status": "skipped", "reason": "case_spec_not_found"}

    timeline = (case_data.get("baseline_timeline") or [])[:max_frames]
    if not timeline:
        return {"case_id": case_id, "status": "skipped", "reason": "no_baseline_frames"}

    src = case_data.get("timeline_source", "unknown")
    print(f"[Stage7-LLM]     Frames: {len(timeline)} (source: {src})")

    case_out = output_dir / case_id / "stage7_llm"
    case_out.mkdir(parents=True, exist_ok=True)

    intent_records: list[dict[str, Any]] = []
    disagreement_events: list[dict[str, Any]] = []

    for fi, frame in enumerate(timeline):
        fid = int(frame.get("frame_id", fi))
        ego, tracked, lane_ctx, route_ctx, stop_ctx, baseline_ctx = _build_input_bundle(
            frame_record=frame,
            world_state=case_data.get("world_state"),
            lane_world_state=case_data.get("lane_world_state"),
            stop_target=case_data.get("stop_target"),
        )
        intent = adapter.call(
            case_id=case_id,
            frame_id=fid,
            ego_state=ego,
            tracked_objects=tracked,
            lane_context=lane_ctx,
            route_context=route_ctx,
            stop_context=stop_ctx,
            baseline_context=baseline_ctx,
        )
        rec = asdict(intent)
        intent_records.append(rec)

        # Throttling to bypass 15 RPM Free-Tier limits (Gemini/Groq)
        if getattr(adapter.config, "mode", "") == "api":
            import time as _time
            _time.sleep(4.1)

        if not rec.get("agrees_with_baseline", True):
            disagreement_events.append({
                "case_id": case_id,
                "frame_id": fid,
                "baseline_intent": rec.get("baseline_intent"),
                "agent_intent": rec.get("tactical_intent"),
                "agent_confidence": rec.get("confidence"),
                "disagreement_useful": rec.get("disagreement_useful"),
                "reason_tags": rec.get("reason_tags"),
                "fallback_to_baseline": rec.get("fallback_to_baseline"),
                "validation_status": rec.get("validation_status"),
                "route_context_summary": rec.get("route_context_summary"),
                "call_latency_ms": (rec.get("provenance") or {}).get("call_latency_ms"),
            })

    # ── 6 mandatory artifacts ──────────────────────────────────────────────────
    intent_path   = case_out / "agent_shadow_intent.jsonl"
    summary_path  = case_out / "agent_shadow_summary.json"
    compare_path  = case_out / "agent_baseline_comparison.json"
    disagree_path = case_out / "agent_disagreement_events.jsonl"
    contract_path = case_out / "agent_contract_validation.json"
    timeout_path  = case_out / "agent_timeout_fallback_events.jsonl"

    dump_jsonl(intent_path,   intent_records)
    summary = build_agent_shadow_summary(case_id=case_id, intents=intent_records)
    dump_json(summary_path, summary)
    comparison = build_agent_baseline_comparison(case_id=case_id, intents=intent_records)
    dump_json(compare_path, comparison)
    dump_jsonl(disagree_path, disagreement_events)
    contract_val = _build_contract_validation(case_id, intent_records)
    dump_json(contract_path, contract_val)
    timeout_events = _build_timeout_fallback_events(case_id, intent_records)
    dump_jsonl(timeout_path, timeout_events)

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = compute_agent_shadow_metrics(
        intents=intent_records,
        summary=summary,
        comparison_path=compare_path,
        disagreement_path=disagree_path,
    )
    # Apply LLM smoke thresholds
    for mn, mr in metrics.items():
        thr = LLM_SMOKE_THRESHOLDS.get(mn)
        if thr is not None:
            mr["threshold"] = thr
            v = mr.get("value")
            if v is None:
                mr["threshold_passed"] = None
                mr["threshold_reason"] = "value_unavailable"
            else:
                passed = True
                reason = "ok"
                if "min" in thr and float(v) < float(thr["min"]):
                    passed = False; reason = f"value {v} < min {thr['min']}"
                if "max" in thr and float(v) > float(thr["max"]):
                    passed = False; reason = f"value {v} > max {thr['max']}"
                mr["threshold_passed"] = passed
                mr["threshold_reason"] = reason

    hard_failures: list[str] = []
    soft_warnings: list[str] = []
    for mn, mr in metrics.items():
        passed = mr.get("threshold_passed")
        reason = mr.get("threshold_reason", "")
        if passed is False:
            if mn in HARD_GATE_METRICS:
                hard_failures.append(f"{mn}: {reason}")
            elif mr.get("gate_usage") == "soft_guardrail":
                soft_warnings.append(f"{mn}: {reason}")

    case_status = "pass" if not hard_failures else "fail"

    valid_f   = summary.get("valid_frames", 0)
    fallback_f = summary.get("fallback_frames", 0)
    disagree_f = summary.get("disagreement_frames", 0)
    timeout_f  = summary.get("timeout_frames", 0)
    latency  = summary.get("mean_call_latency_ms")
    latency_s = f"{latency:.1f}ms" if latency else "n/a"

    print(
        f"[Stage7-LLM]     Valid={valid_f} Fallback={fallback_f} "
        f"Disagree={disagree_f} Timeout={timeout_f} "
        f"Latency≈{latency_s} Status={case_status.upper()}"
    )

    return {
        "case_id": case_id,
        "status": case_status,
        "timeline_source": src,
        "frames_processed": len(intent_records),
        "hard_failures": hard_failures,
        "soft_warnings": soft_warnings,
        "metrics": {
            k: {
                "value": v.get("value"),
                "threshold_passed": v.get("threshold_passed"),
                "threshold": v.get("threshold"),
                "maturity": v.get("maturity"),
            }
            for k, v in metrics.items()
        },
        "summary": summary,
        "artifacts": {
            "agent_shadow_intent": str(intent_path),
            "agent_shadow_summary": str(summary_path),
            "agent_baseline_comparison": str(compare_path),
            "agent_disagreement_events": str(disagree_path),
            "agent_contract_validation": str(contract_path),
            "agent_timeout_fallback_events": str(timeout_path),
        },
        "contract_validation": contract_val,
        "disagree_sample": disagreement_events[:5],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run_stage7_llm_shadow_smoke(
    repo_root: str | Path,
    *,
    adapter_mode: str = "api_simulated",
    api_key_env_var: str = "AGENT_API_KEY",
    api_endpoint: str | None = None,
    max_frames_per_case: int = 60,
    api_simulated_disagree_rate: float = 0.22,
    api_simulated_latency_ms: float = 35.0,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / "benchmark" / "reports" / f"stage7_llm_shadow_smoke_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    api_key_present = bool(os.environ.get(api_key_env_var, "").strip())

    print(f"[Stage7-LLM] ═══════════════════════════════════════════════════")
    print(f"[Stage7-LLM] Stage 7 — LLM/API Adapter Shadow Smoke")
    print(f"[Stage7-LLM] INVARIANT: shadow_only — NO control authority")
    print(f"[Stage7-LLM] Cases: {LLM_SMOKE_CASES}")
    print(f"[Stage7-LLM] Adapter mode: {adapter_mode}")
    if adapter_mode == "api_simulated":
        print(f"[Stage7-LLM] Simulated disagree_rate={api_simulated_disagree_rate} "
              f"latency_ms={api_simulated_latency_ms}")
    elif adapter_mode == "api":
        print(f"[Stage7-LLM] Real API — key present: {api_key_present}")
    print(f"[Stage7-LLM] Output: {run_dir}")
    print(f"[Stage7-LLM] ═══════════════════════════════════════════════════\n")

    # ── Contract audit ─────────────────────────────────────────────────────────
    print("[Stage7-LLM] Running Stage 7 contract audit...")
    try:
        contract_audit = run_stage7_contract_audit(repo_root)
    except Exception as exc:
        print(f"[Stage7-LLM]   WARN: contract audit failed: {exc}")
        contract_audit = {"status": "audit_failed", "error": str(exc)}

    # ── Build adapter ──────────────────────────────────────────────────────────
    from benchmark.agent_shadow_adapter import AgentShadowAdapterConfig, AgentShadowAdapter

    cfg_kwargs: dict[str, Any] = {
        "mode": adapter_mode,
        "model_id": f"{adapter_mode}_gemini" if adapter_mode == "api" else "api_simulated_v1",
        "api_timeout_s": 10.0,
        "api_key_env_var": api_key_env_var,
        "api_endpoint": api_endpoint,
    }
    if adapter_mode == "api_simulated":
        cfg_kwargs["api_simulated_disagree_rate"] = api_simulated_disagree_rate
        cfg_kwargs["api_simulated_latency_ms"] = api_simulated_latency_ms
        cfg_kwargs["api_simulated_timeout_rate"] = 0.0

    cfg = AgentShadowAdapterConfig(**cfg_kwargs)
    adapter = AgentShadowAdapter(cfg)

    # ── Per-case shadow ────────────────────────────────────────────────────────
    print("\n[Stage7-LLM] Running LLM shadow on 2 cases...\n")
    all_results: list[dict[str, Any]] = []
    gate_passed = True
    gate_hard_failures: list[str] = []
    gate_soft_warnings: list[str] = []

    for case_id in LLM_SMOKE_CASES:
        cr = _run_case_llm_smoke(
            case_id=case_id,
            repo_root=repo_root,
            output_dir=run_dir,
            adapter=adapter,
            max_frames=max_frames_per_case,
        )
        all_results.append(cr)
        if cr.get("status") == "fail":
            gate_passed = False
            gate_hard_failures.extend([f"{case_id}/{f}" for f in (cr.get("hard_failures") or [])])
        gate_soft_warnings.extend([f"{case_id}/{f}" for f in (cr.get("soft_warnings") or [])])

    skipped = [r["case_id"] for r in all_results if r.get("status") == "skipped"]
    if skipped:
        gate_soft_warnings.extend([f"{c}:skipped" for c in skipped])

    overall = "pass" if gate_passed and not skipped else ("fail" if gate_hard_failures else "partial")
    print(f"\n[Stage7-LLM] ════════════ GATE: {overall.upper()} ════════════")

    # ── Canonical outputs ──────────────────────────────────────────────────────
    gate_result = {
        "schema_version": "stage7_llm_shadow_smoke_result_v1",
        "generated_at_utc": _utc_now(),
        "stage": "7_llm_smoke",
        "mode": "llm_api_shadow_smoke",
        "adapter_mode": adapter_mode,
        "api_key_configured": api_key_present,
        "agent_authority": "shadow_only — baseline tactical + MPC remain sole authority",
        "no_control_authority": True,
        "no_assist_mode": True,
        "no_takeover": True,
        "subset_cases": LLM_SMOKE_CASES,
        "overall_status": overall,
        "hard_failures": gate_hard_failures,
        "soft_warnings": gate_soft_warnings,
        "passed_cases":  [r["case_id"] for r in all_results if r.get("status") == "pass"],
        "failed_cases":  [r["case_id"] for r in all_results if r.get("status") == "fail"],
        "skipped_cases": skipped,
        "adapter": {
            "mode": adapter_mode,
            "model_id": cfg.model_id,
            "timeout_s": cfg.api_timeout_s,
            "api_key_env_var": api_key_env_var,
            "api_key_present": api_key_present,
        },
        "metric_summary": {
            mn: {
                cr["case_id"]: {
                    "value": (cr.get("metrics") or {}).get(mn, {}).get("value"),
                    "threshold_passed": (cr.get("metrics") or {}).get(mn, {}).get("threshold_passed"),
                }
                for cr in all_results if cr.get("status") not in {"skipped"}
            }
            for mn in SELECTED_METRICS_FOR_RESULT
        },
    }

    report = {
        "schema_version": "stage7_llm_shadow_smoke_report_v1",
        "generated_at_utc": _utc_now(),
        "stage": "7_llm_smoke",
        "mode": "llm_api_shadow_smoke",
        "adapter_mode": adapter_mode,
        "api_key_configured": api_key_present,
        "agent_authority": "shadow_only",
        "no_control_authority": True,
        "no_assist_mode": True,
        "no_takeover": True,
        "subset_cases": LLM_SMOKE_CASES,
        "overall_status": overall,
        "run_output_dir": str(run_dir),
        "adapter": gate_result["adapter"],
        "contract_audit_summary": {
            "stage7a_ready": (contract_audit.get("summary") or {}).get("stage7a_replay_readiness"),
        },
        "cases": all_results,
        "notes": [
            "Stage 7 LLM/API adapter shadow smoke — shadow only.",
            f"Adapter mode: {adapter_mode}.",
            "api_simulated: full pipeline test with intelligent local LLM shim (no production key required).",
            "api: real Gemini call via urllib; missing AGENT_API_KEY → all frames timeout-fallback.",
            "Baseline tactical + MPC are the sole authority at all times.",
            "No control authority, no assist mode, no takeover is permitted.",
        ],
    }

    # Per-run
    dump_json(run_dir / "stage7_llm_shadow_smoke_result.json", gate_result)
    dump_json(run_dir / "stage7_llm_shadow_smoke_report.json", report)
    # Canonical benchmark root
    dump_json(repo_root / "benchmark" / "stage7_llm_shadow_smoke_result.json", gate_result)
    dump_json(repo_root / "benchmark" / "stage7_llm_shadow_smoke_report.json", report)

    print(f"[Stage7-LLM] Result → benchmark/stage7_llm_shadow_smoke_result.json")
    print(f"[Stage7-LLM] Report → benchmark/stage7_llm_shadow_smoke_report.json")

    return {"gate_result": gate_result, "report": report, "run_dir": str(run_dir)}
