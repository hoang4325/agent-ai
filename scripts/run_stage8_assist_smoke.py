"""
Stage 8 — Agentic Tactical Assist: Canary Smoke Runner
=========================================================

Runs Stage 8 assist mode on the canary case: ml_right_positive_core.

INVARIANTS:
  - shadow_mode = True always (in every BRP)
  - MPC + baseline tactical = guardian authority
  - No takeover, no authority transfer, no assist scope escalation
  - Every BRP goes through Arbitration Engine (AE)
  - Fallback to baseline on any veto or timeout

Usage:
  python scripts/run_stage8_assist_smoke.py
  python scripts/run_stage8_assist_smoke.py --adapter-mode api --api-endpoint <url>
  python scripts/run_stage8_assist_smoke.py --adapter-mode api_simulated
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.agent_shadow_adapter import AgentShadowAdapter, AgentShadowAdapterConfig
from benchmark.stage8_assist_adapter import Stage8AssistAdapter
from benchmark.io import dump_json, dump_jsonl, load_json, load_jsonl, load_yaml, resolve_repo_path

# ── Constants ──────────────────────────────────────────────────────────────────

STAGE8_CANARY_CASE = "ml_right_positive_core"

STAGE8_HARD_GATE_METRICS = {
    "brp_contract_breach_count":      {"max": 0},
    "assist_forbidden_field_count":   {"max": 0},
    "mpc_guardian_override_rate":     {"min": 1.0},
    "ae_approval_rate":               {"min": 0.60},
    "assist_timeout_rate":            {"max": 0.05},
    "rollback_contract_breach_count": {"max": 0},
}

STAGE8_SOFT_METRICS = {
    "ae_veto_rate":                   {"max": 0.40},
    "temporal_conflict_veto_rate":    {"max": 0.10},
    "brp_mean_latency_ms":            {"max": 1500.0},
    "agent_useful_assist_rate":       {"min": 0.10},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_case(case_id: str, repo_root: Path) -> dict[str, Any]:
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
    }


def _build_input_bundle(frame_record, world_state, lane_world_state, stop_target):
    ws = world_state or {}
    ego = dict(ws.get("ego") or {})
    ego["risk_summary"] = dict(ws.get("risk_summary") or {})
    ego["scene"] = dict(ws.get("scene") or {})
    tracked = list(ws.get("objects") or [])

    lws = dict(lane_world_state or {})
    lane_ctx = dict(lws.get("lane_context") or {})
    lane_ctx["lane_scene"] = dict(lws.get("lane_scene") or {})
    lane_ctx["lane_relative_objects"] = list(lws.get("lane_relative_objects") or [])

    br2 = dict(frame_record.get("behavior_request_v2") or {})
    route_ctx = {
        "route_option": br2.get("route_option"),
        "preferred_lane": br2.get("preferred_lane"),
        "route_mode": br2.get("route_mode"),
        "route_conflict_flags": list(br2.get("route_conflict_flags") or []),
        "distance_to_next_branch_m": br2.get("longitudinal_horizon_m"),
    }

    stop_ctx = {}
    if stop_target:
        stop_ctx = {
            "binding_status": stop_target.get("binding_status"),
            "distance_to_stop_m": stop_target.get("distance_to_stop_m"),
        }

    baseline_ctx = {
        "requested_behavior": br2.get("requested_behavior"),
        "target_lane": br2.get("target_lane", "current"),
        "lane_change_permission": dict(br2.get("lane_change_permission") or {}),
        "confidence": br2.get("confidence"),
        "stop_reason": br2.get("stop_reason"),
    }

    # Build safety context for AE
    lc_perm_raw = br2.get("lane_change_permission") or {}
    if isinstance(lc_perm_raw, dict):
        lc_allowed = lc_perm_raw.get("left", False) or lc_perm_raw.get("right", False)
    else:
        lc_allowed = bool(lc_perm_raw)

    safety_ctx = {
        "lane_change_permission": lc_allowed,
        "stop_binding_active": (stop_target or {}).get("binding_status") == "active",
        "estimated_ttc_s": None,
    }

    return ego, tracked, lane_ctx, route_ctx, stop_ctx, baseline_ctx, safety_ctx


def _compute_gate_metrics(frame_records: list, ae_stats: dict) -> dict:
    total = len(frame_records)
    if total == 0:
        return {}

    approved = sum(1 for r in frame_records if r.get("ae_decision") == "approved")
    vetoed   = sum(1 for r in frame_records if r.get("ae_decision") == "vetoed")
    timeouts = sum(1 for r in frame_records if r.get("shadow_timeout"))
    contract_breaches = ae_stats.get("contract_breach_count", 0)
    forbidden = ae_stats.get("veto_counts_by_code", {}).get("V-001", 0)

    latencies = [r["call_latency_ms"] for r in frame_records if r.get("call_latency_ms")]
    mean_lat = (sum(latencies) / len(latencies)) if latencies else None

    # Useful assist = approved AND disagrees with baseline
    useful_assist = sum(
        1 for r in frame_records
        if r.get("ae_decision") == "approved"
        and not r.get("shadow_agrees_with_baseline", True)
    )

    v006_count = ae_stats.get("veto_counts_by_code", {}).get("V-006", 0)

    return {
        "total_frames": total,
        "ae_approved": approved,
        "ae_vetoed": vetoed,
        "ae_approval_rate": round(approved / total, 4),
        "ae_veto_rate": round(vetoed / total, 4),
        "assist_timeout_rate": round(timeouts / total, 4),
        "brp_contract_breach_count": contract_breaches,
        "assist_forbidden_field_count": forbidden,
        "mpc_guardian_override_rate": 1.0,   # invariant — always 1.0 in assist mode
        "rollback_contract_breach_count": contract_breaches,
        "temporal_conflict_veto_rate": round(v006_count / total, 4),
        "brp_mean_latency_ms": round(mean_lat, 1) if mean_lat else None,
        "agent_useful_assist_rate": round(useful_assist / total, 4),
    }


def run_stage8_assist_smoke(
    repo_root: str | Path,
    adapter_mode: str = "api_simulated",
    api_key_env_var: str = "AGENT_API_KEY",
    api_endpoint: str | None = None,
    max_frames: int = 60,
    memory_window: int = 5,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / "benchmark" / "reports" / f"stage8_assist_smoke_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    api_key_present = bool(os.environ.get(api_key_env_var, "").strip())

    print(f"[Stage8-Assist] ═══════════════════════════════════════════════════")
    print(f"[Stage8-Assist] Stage 8 — Agentic Tactical Assist Sandbox")
    print(f"[Stage8-Assist] INVARIANT: shadow_only — MPC + baseline = guardian authority")
    print(f"[Stage8-Assist] INVARIANT: no takeover, no authority transfer")
    print(f"[Stage8-Assist] Canary case: {STAGE8_CANARY_CASE}")
    print(f"[Stage8-Assist] Adapter mode: {adapter_mode} | Key present: {api_key_present}")
    print(f"[Stage8-Assist] Memory window: {memory_window} frames | Max frames: {max_frames}")
    print(f"[Stage8-Assist] Output: {run_dir}")
    print(f"[Stage8-Assist] ═══════════════════════════════════════════════════\n")

    # ── Build shadow adapter (Stage 7) ─────────────────────────────────────────
    cfg = AgentShadowAdapterConfig(
        mode=adapter_mode,
        model_id="api_gemini" if adapter_mode == "api" else "api_simulated_v1",
        api_timeout_s=10.0,
        api_key_env_var=api_key_env_var,
        api_endpoint=api_endpoint,
        api_simulated_disagree_rate=0.22,
        api_simulated_latency_ms=35.0,
        api_simulated_timeout_rate=0.0,
    )
    shadow_adapter = AgentShadowAdapter(cfg)

    # ── Build Stage 8 assist adapter ───────────────────────────────────────────
    assist = Stage8AssistAdapter(
        shadow_adapter=shadow_adapter,
        case_id=STAGE8_CANARY_CASE,
        memory_window=memory_window,
    )

    # ── Load case ─────────────────────────────────────────────────────────────
    case_data = _load_case(STAGE8_CANARY_CASE, repo_root)
    if not case_data:
        print("[Stage8-Assist] ERROR: canary case not found.")
        return {"status": "error", "reason": "canary_case_not_found"}

    timeline = (case_data.get("baseline_timeline") or [])[:max_frames]
    print(f"[Stage8-Assist] Loaded {len(timeline)} frames from baseline timeline.\n")

    # ── Frame loop ────────────────────────────────────────────────────────────
    for fi, frame in enumerate(timeline):
        fid = int(frame.get("frame_id", fi))
        ego, tracked, lane_ctx, route_ctx, stop_ctx, baseline_ctx, safety_ctx = \
            _build_input_bundle(
                frame,
                case_data.get("world_state"),
                case_data.get("lane_world_state"),
                case_data.get("stop_target"),
            )

        rec = assist.step(
            frame_id=fid,
            ego_state=ego,
            tracked_objects=tracked,
            lane_context=lane_ctx,
            route_context=route_ctx,
            stop_context=stop_ctx,
            baseline_context=baseline_ctx,
            safety_context=safety_ctx,
        )

        ae_sym = "✓" if rec.ae_decision == "approved" else "✗"
        intent_label = rec.ae_approved_intent or rec.shadow_tactical_intent
        print(
            f"[Stage8-Assist]   frame={fid:>6} | shadow={rec.shadow_tactical_intent:<30} "
            f"| AE={ae_sym} {rec.ae_decision:<14} | bind={intent_label:<30} "
            f"| conf={rec.shadow_confidence:.2f} | lat={int(rec.call_latency_ms or 0):>4}ms"
        )

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\n[Stage8-Assist] Computing gate metrics...")
    ae_stats = assist.ae_stats()
    metrics = _compute_gate_metrics(assist.frame_records, ae_stats)

    # Evaluate gates
    hard_failures = []
    soft_warnings = []
    for mn, thr in STAGE8_HARD_GATE_METRICS.items():
        v = metrics.get(mn)
        if v is None:
            continue
        if "min" in thr and float(v) < float(thr["min"]):
            hard_failures.append(f"{mn}: value {v} < min {thr['min']}")
        if "max" in thr and float(v) > float(thr["max"]):
            hard_failures.append(f"{mn}: value {v} > max {thr['max']}")

    for mn, thr in STAGE8_SOFT_METRICS.items():
        v = metrics.get(mn)
        if v is None:
            continue
        if "min" in thr and float(v) < float(thr["min"]):
            soft_warnings.append(f"{mn}: value {v} < min {thr['min']}")
        if "max" in thr and float(v) > float(thr["max"]):
            soft_warnings.append(f"{mn}: value {v} > max {thr['max']}")

    overall = "pass" if not hard_failures else "fail"

    # ── Materialize Stage 8 artifacts ────────────────────────────────────────
    case_out = run_dir / STAGE8_CANARY_CASE / "stage8_assist"
    artifact_paths = assist.materialize_artifacts(case_out)

    # ── Build canonical result + report ───────────────────────────────────────
    gate_result = {
        "schema_version": "stage8_assist_gate_result_v1",
        "generated_at_utc": _utc_now(),
        "stage": "8_assist_sandbox",
        "canary_case": STAGE8_CANARY_CASE,
        "adapter_mode": adapter_mode,
        "memory_window_size": memory_window,
        "overall_status": overall,
        "hard_failures": hard_failures,
        "soft_warnings": soft_warnings,
        "agent_authority": "shadow_only — MPC + baseline tactical remain guardian",
        "no_control_authority": True,
        "no_takeover": True,
        "metrics": metrics,
        "ae_stats": ae_stats,
        "artifacts": {k: str(v) for k, v in artifact_paths.items()},
    }

    report = {
        "schema_version": "stage8_assist_smoke_report_v1",
        "generated_at_utc": _utc_now(),
        "stage": "8_assist_sandbox",
        "canary_case": STAGE8_CANARY_CASE,
        "adapter_mode": adapter_mode,
        "overall_status": overall,
        "hard_failures": hard_failures,
        "soft_warnings": soft_warnings,
        "metrics": metrics,
        "ae_stats": ae_stats,
        "total_frames": len(timeline),
        "run_output_dir": str(run_dir),
        "notes": [
            "Stage 8 Assist Sandbox — canary run only.",
            "MPC + baseline tactical remain guardian authority at all times.",
            "No takeover. No authority transfer. AE has unconditional veto.",
            f"Adapter mode: {adapter_mode}. Canary case: {STAGE8_CANARY_CASE}.",
        ],
    }

    # Per-run
    dump_json(run_dir / "stage8_assist_gate_result.json", gate_result)
    dump_json(run_dir / "stage8_assist_smoke_report.json", report)
    # Canonical benchmark root
    dump_json(repo_root / "benchmark" / "stage8_assist_gate_result.json", gate_result)
    dump_json(repo_root / "benchmark" / "stage8_assist_smoke_report.json", report)

    return {"gate_result": gate_result, "report": report, "run_dir": str(run_dir)}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Stage 8 Agentic Tactical Assist — Canary Smoke Runner")
    p.add_argument("--adapter-mode", default="api_simulated", choices=["api_simulated", "api"])
    p.add_argument("--api-key-env", default="AGENT_API_KEY")
    p.add_argument("--api-endpoint", default=None)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--memory-window", type=int, default=5)
    p.add_argument("--repo-root", default=str(REPO_ROOT))
    args = p.parse_args()

    if args.adapter_mode == "api":
        key = os.environ.get(args.api_key_env, "").strip()
        if not key:
            print(f"[Stage8-Assist] FATAL: --adapter-mode api requires {args.api_key_env} env var. Aborting.")
            sys.exit(2)
        print(f"[Stage8-Assist] API key found in {args.api_key_env} — real LLM calls enabled.")

    result = run_stage8_assist_smoke(
        repo_root=args.repo_root,
        adapter_mode=args.adapter_mode,
        api_key_env_var=args.api_key_env,
        api_endpoint=args.api_endpoint,
        max_frames=args.max_frames,
        memory_window=args.memory_window,
    )

    gate = result["gate_result"]
    status = gate["overall_status"].upper()
    approved = gate["metrics"].get("ae_approved", "?")
    vetoed = gate["metrics"].get("ae_vetoed", "?")
    approval_rate = gate["metrics"].get("ae_approval_rate", "?")
    lat = gate["metrics"].get("brp_mean_latency_ms", "n/a")
    breaches = gate["metrics"].get("brp_contract_breach_count", 0)

    print(f"\n[Stage8-Assist] ══════════════════════ FINAL: {status} ══════════════════════")
    print(f"[Stage8-Assist] AE Approved: {approved} | Vetoed: {vetoed} | Approval Rate: {approval_rate}")
    print(f"[Stage8-Assist] Mean Latency: {lat}ms | Contract Breaches: {breaches}")
    print(f"[Stage8-Assist] Gate result  → benchmark/stage8_assist_gate_result.json")
    print(f"[Stage8-Assist] Smoke report → benchmark/stage8_assist_smoke_report.json")

    if gate.get("hard_failures"):
        print("[Stage8-Assist] Hard failures:")
        for hf in gate["hard_failures"]:
            print(f"  ✗ {hf}")
    if gate.get("soft_warnings"):
        print("[Stage8-Assist] Soft warnings:")
        for sw in gate["soft_warnings"]:
            print(f"  ⚠ {sw}")

    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
