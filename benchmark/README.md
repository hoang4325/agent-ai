# System Benchmark v1

System Benchmark v1 is the first runnable benchmark scaffold for the Agent-AI autonomous-driving stack.

Design principles:
- scenario-driven
- contract-aware
- deterministic where possible
- layered across perception, world model, behavior, execution, and online runtime

This v1 is intentionally artifact-driven:
- it reuses existing Stage 1 -> Stage 3C outputs immediately
- it can report optional Stage 4 runtime metrics if online artifacts are present
- it is a baseline for regression and architecture quality gates, not the final benchmark system

Frozen corpus v1 upgrade:
- `benchmark/frozen_corpus/v1/` now materializes `frozen_case_bundle_v1` bundles from existing case artifacts
- future primary replay input is:
  - scenario manifest / route contract
  - `stage1_frozen/` replayable session
- `reference_outputs/` under each bundle stores frozen Stage2/3A/3B/3C outputs for regression and baseline comparison
- `benchmark/mapping_benchmark_to_frozen_corpus.json` lets the existing runners reuse the corpus without replacing v1/v1.1 flows

Directory layout:
- `benchmark_v1.yaml`: top-level benchmark config
- `metric_registry.yaml`: metric definitions, maturity, and gate usage
- `cases/`: per-scenario benchmark case specs
- `sets/`: named benchmark sets
- `baselines/system_benchmark_v1/`: pinned baseline metadata and metrics
- `reports/`: benchmark run outputs
- `schemas/`: lightweight JSON schemas for config and case files

Current maturity:
- replay-oriented cases are runnable now
- some stress and route/session cases are kept as `planned` because clean artifacts are not yet available
- online/runtime metrics are present in the registry, but most are `soft_guardrail` or `diagnostic_only` until Stage 4 artifacts become stable and case-specific

v1.1 semantic/binding hardening (backward-compatible):
- benchmark runner can now auto-resolve or auto-generate semantic artifacts per case:
	- `ground_truth_actor_roles.json`
	- `critical_actor_binding.jsonl`
	- `behavior_arbitration_events.jsonl`
	- `route_session_binding.json`
- when semantic artifacts are missing, metrics are explicitly marked as `metric_skipped` or `metric_degraded`
- existing v1 case specs and legacy artifacts remain runnable without migration blockers

Benchmark mode taxonomy:
- `replay_regression`: evaluate existing artifacts only
- `scenario_replay`: run Stage2->3C replay chain from scenario/session inputs, then evaluate
- `online_e2e`: run live integrated Stage1->4 online loop (via Stage4 orchestrator), then evaluate

Stage 5A gate posture:
- `scripts/run_stage5a_gate.py` is the authoritative Stage 5A entrypoint
- Stage 5A is now driven by `frozen_corpus + scenario_replay`, not replay-only golden reports
- threshold authority is:
  - primary: `case_spec.metric_thresholds`
  - fallback: `benchmark_v1.default_metric_thresholds`
- `benchmark/stage5a_metric_pack.json` is taxonomy/advisory, not the runner truth source

Stage 5B gate posture:
- `scripts/run_stage5b_gate.py` is the authoritative Stage 5B entrypoint
- Stage 5B reuses the same `frozen_corpus + scenario_replay` backbone, then re-evaluates current-run bundles through a Stage 5B runtime case overlay
- Stage 5B adds planner / execution quality artifacts under every Stage 3C run bundle:
  - `lane_change_events_enriched.jsonl`
  - `stop_events.jsonl`
  - `trajectory_trace.jsonl`
  - `control_trace.jsonl`
  - `lane_change_phase_trace.jsonl`
  - `stop_target.json`
  - `planner_quality_summary.json`
- current Stage 5B hard gate metrics are:
  - `lane_change_completion_rate`
  - `lane_change_completion_validity`
  - `stop_hold_stability`
- Stage 5B soft guardrails are reported but not promoted to hard fail until artifact support is cleaner:
  - `lane_change_completion_time_s`
  - `lane_change_abort_rate`
  - `heading_settle_time_s`
  - `lateral_oscillation_rate`
  - `trajectory_smoothness_proxy`
  - `stop_overshoot_distance_m`
- Stage 5B diagnostic-only metrics remain non-gating:
  - `stop_final_error_m`
  - `control_jerk_proxy`
  - `planner_execution_latency_ms`

Stage 6 preflight posture:
- `scripts/run_stage6_preflight.py` is the contract-readiness entrypoint before any MPC shadow work
- this step does not add MPC; it hardens the artifacts and benchmark surface for:
  - `stop_target`
  - `planner_control_latency`
  - `fallback_events`
- new Stage 3C / Stage 4 current-run artifacts now include:
  - `planner_control_latency.jsonl`
  - `fallback_events.jsonl`
  - upgraded `stop_target.json`
- current maturity:
  - hard-ready contract metrics:
    - `stop_target_defined_rate`
    - `fallback_reason_coverage`
  - soft guardrail:
    - `planner_execution_latency_ms`
    - `decision_to_execution_latency_ms`
    - `over_budget_rate`
    - `fallback_activation_rate`
    - `fallback_missing_when_required_count`
    - `stop_overshoot_distance_m`
  - diagnostic only:
    - `stop_final_error_m`
    - `fallback_takeover_latency_ms`

Stage 6 revalidation posture:
- `scripts/run_stage6_preflight_revalidation.py` now replays a minimal shadow subset with:
  - `scenario_replay` current-run evidence
  - `online_e2e` Stage 4 evidence using `stage4_online_external_watch`
- `stage4_online_external_watch` is intentionally diagnostic:
  - it keeps Stage 4 live/current-run
  - it reuses `stage1_frozen/session` instead of requiring Docker live watch
  - it is for contract evidence, not a replacement for full live Stage 1 online bring-up
- `scripts/run_stage4_stop_materialization_revalidation.py` is the stop-focused Stage 4 exact-binding probe:
  - reruns `stop_follow_ambiguity_core` online with `stage1_frozen/session`
  - emits `critical_actor_materialization.json`, `stop_target_trace.jsonl`, `stop_binding_summary.json`
  - updates `benchmark/stage4_stop_materialization_audit.json` and `benchmark/stage6_stop_target_readiness.json`
  - this probe is considered `partial` unless:
    - blocker binding is `exact`
    - `stop_final_error_m` is directly available
    - `alignment_consistent = true`
- current stop-target posture after the actor-materialization patch:
  - exact runtime blocker binding can now be materialized on the stop-focused subset
  - `stop_final_error_m` can be computed on that subset
  - but `stop_final_error_m` must stay out of soft/hard gating while `alignment_consistent = false`, because `external_watch` can still replay a late frozen stop state against an earlier online ego pose

Stage 6 shadow rollout posture:
- `scripts/run_stage6_shadow_gate.py` opens the first MPC shadow rollout on exactly:
  - `ml_right_positive_core`
  - `stop_follow_ambiguity_core`
- this is a proposal-only rollout:
  - baseline execution/control remains the only authority
  - shadow artifacts are emitted into the current-run `stage3c/` directory
  - no takeover path is enabled
- shadow artifact contract now includes:
  - `mpc_shadow_proposal.jsonl`
  - `mpc_shadow_summary.json`
  - `shadow_baseline_comparison.json`
  - `shadow_disagreement_events.jsonl`
- current Stage 6 shadow maturity:
  - hard gate:
    - `artifact_completeness`
    - `shadow_output_contract_completeness`
    - `shadow_feasibility_rate`
  - soft guardrail:
    - `shadow_solver_timeout_rate`
    - `shadow_planner_execution_latency_ms`
    - `shadow_baseline_disagreement_rate`
    - `shadow_fallback_recommendation_rate`
    - `shadow_stop_final_error_m`
    - `shadow_stop_overshoot_distance_m`
    - `shadow_lateral_oscillation_rate`
    - `shadow_trajectory_smoothness_proxy`
    - `shadow_control_jerk_proxy`
  - diagnostic only:
    - `shadow_lane_change_completion_time_s`
    - `shadow_heading_settle_time_s`
- shadow reports now expose per-metric support provenance:
  - `proposal_direct`
  - `proposal_direct_with_heuristic_stop_target`
  - `proposal_direct_solver_timing`
  - `counterfactual_from_shadow_path`
  - `unavailable`
- use `shadow_metric_support` and `soft_guardrail_candidate_metrics` inside `mpc_shadow_summary.json` / `shadow_baseline_comparison.json` to decide which metrics are trustworthy enough for trend watching
- current rollout posture after the first shadow gate run:
  - the promoted 2-case subset is benchmarkable in shadow mode
  - shadow output contract is now backed by an OSQP-based minimal MPC shadow solver, not the original baseline-anchored proxy generator
  - hard gate is clean and the current 2-case shadow gate is `pass`
  - baseline remains the only execution authority; shadow stays proposal-only
  - do not expand beyond the 2 promoted cases until the minimal OSQP backend is validated against a fuller MPC solver

Stage 6 online shadow smoke posture:
- `scripts/run_stage6_online_shadow_smoke.py` is the online smoke supplement for the same promoted 2-case subset:
  - `ml_right_positive_core`
  - `stop_follow_ambiguity_core`
- Stage 4 now emits online shadow artifacts directly into the current-run bundle:
  - `mpc_shadow_proposal.jsonl`
  - `mpc_shadow_summary.json`
  - `shadow_baseline_comparison.json`
  - `shadow_disagreement_events.jsonl`
- current online shadow smoke posture:
  - `benchmark/stage6_online_shadow_smoke_result.json` is now `pass`
  - shadow contract audit is `present`
  - baseline remains the only execution authority; shadow stays proposal-only
  - online smoke only gates contract completeness, feasibility, disagreement/fallback behavior, and smoke-tier latency
  - proposal-direct quality metrics are now surfaced separately from counterfactual ones
  - even for proposal-direct metrics, shadow still must not be used to claim MPC superiority as executed control because baseline remains the only controller

Stage 6 online shadow stability posture:
- `scripts/run_stage6_online_shadow_stability.py` runs repeated online shadow smoke passes on the same 2-case subset and aggregates repeatability:
  - runtime success rate per case
  - gate pass rate per case
  - latency/disagreement/fallback variance
- current truth after runtime hardening:
  - the 2-case online shadow smoke path is runnable and repeatable
  - the current 3-run campaign is `ready`
  - CARLA readiness retry in Stage 4 removed the restart-window `get_world()` timeout that previously caused intermittent run failures
  - buffered shadow artifact writing plus new latency breakdown artifacts show bookkeeping overhead is tiny compared with solver time, so online shadow latency jitter is primarily solver/runtime-side, not JSONL write overhead
- this campaign intentionally restarts CARLA between iterations so CARLA reuse crashes do not masquerade as shadow-path instability

Stage 6 online shadow golden posture:
- `scripts/run_stage6_online_shadow_golden_gate.py` promotes the same 2-case subset into a pinned online shadow golden regression gate
- golden baseline pinning now uses the ready repeatability campaign envelope instead of a single smoke snapshot, so the gate compares against the observed stable runtime band rather than one lucky run
- online current-run Stage 3C bundles now also carry realized short-horizon comparison artifacts:
  - `shadow_realized_outcome_trace.jsonl`
  - `shadow_realized_outcome_summary.json`
- realized short-horizon comparison stays proposal-only and counterfactual-safe:
  - it compares each shadow proposal against the realized future baseline window from the same run
  - it does **not** imply shadow takeover or executed-control authority
  - `shadow_realized_support_rate` now uses `eligible_proposals` only, so tail-truncated proposals at the end of a run are excluded instead of being counted as false misses
- current truth:
  - `benchmark/stage6_online_shadow_golden_result.json` is `pass`
  - the online shadow golden gate stays proposal-only and subset-limited
  - baseline remains the only execution authority
- latency semantics are now explicit:
  - `shadow_planner_execution_latency_ms` is the direct OSQP `solve()` time used by the shadow contract/gate
  - `shadow_runtime_latency_summary.json` carries:
    - `mean_problem_build_latency_ms`
    - `mean_solver_setup_latency_ms`
    - `mean_planner_total_compute_latency_ms`
    - `mean_shadow_tick_runtime_ms`
  - use the summary artifact, not the gate metric alone, when diagnosing Stage 4 runtime cost or deciding whether the solver path is ready to expand
  - realized short-horizon metrics are currently usable as:
    - soft guardrail:
      - `shadow_realized_support_rate`
      - `shadow_realized_behavior_match_rate`
    - diagnostic only:
      - `shadow_realized_speed_mae_mps`
      - `shadow_realized_progress_mae_m`
      - `shadow_realized_stop_distance_mae_m`
  - the current optimized posture is materially cleaner:
    - `ml_right_positive_core` latest stable run:
      - solve latency `~0.10 ms`
      - total compute latency `~0.70 ms`
      - total shadow tick runtime `~0.90 ms`
    - `stop_follow_ambiguity_core` latest stable run:
      - solve latency `~0.11 ms`
      - total compute latency `~1.06 ms`
      - total shadow tick runtime `~1.18 ms`

With frozen corpus mapping enabled:
- `replay_regression` resolves Stage1/2/3A/3B/3C reference artifacts from `benchmark/frozen_corpus/v1/` when a case bundle is available
- `scenario_replay` resolves `scenario_manifest + stage1_frozen/session` from the corpus for cases that are `corpus_ready`
- `online_e2e` keeps using live/current-run artifacts for evaluation, but may read corpus scenario/reference contracts for case setup and expectations

E2E orchestrator entrypoint:

```powershell
python scripts\run_system_benchmark_e2e.py --config benchmark\benchmark_v1.yaml --set online_e2e_smoke --benchmark-mode online_e2e --execute-stages
python scripts\run_system_benchmark_e2e.py --config benchmark\benchmark_v1.yaml --set golden --benchmark-mode replay_regression
```

E2E orchestration outputs:
- `benchmark_execution_trace.jsonl`
- per-case `artifact_provenance.json`
- `online_e2e_case_result.json`

Stage4 runtime bring-up diagnostics:

```powershell
python scripts\run_stage4_infra_smoke.py --output-dir outputs\stage4_infra_smoke
python scripts\stage4_runtime_audit.py --output outputs\stage4_infra_smoke\stage4_runtime_audit.json --strict
```

- `stage4_runtime_audit.json` reports Python tag, discovered CARLA dist artifacts, bootstrap path changes, import/connect checks.
- `stage4_infra_smoke_summary.json` is a one-file pass/fail readiness signal for online Stage4 bring-up.
- If Stage4 must run in a different interpreter than benchmark evaluation, set `stage4_python_executable` in case spec (used by `stage4_online_default` profile).

Online stability audit workflow:

```powershell
python scripts\online_smoke_stability.py --reports-root benchmark\reports --output-root benchmark\reports\online_stability
python scripts\online_metric_stability.py --stability-report benchmark\reports\online_stability\online_smoke_stability_report.json --output benchmark\reports\online_stability\online_metric_stability_registry.json
python scripts\online_baseline_repin.py --stability-report benchmark\reports\online_stability\online_smoke_stability_report.json --metric-stability-registry benchmark\reports\online_stability\online_metric_stability_registry.json --baseline-dir benchmark\baselines\system_benchmark_v1_online --decision-output benchmark\reports\online_stability\online_baseline_decision.json
```

Promotion policy:
- keep `online_e2e_smoke` as smoke until repeatability evidence is present
- promote only stable/acceptable cases into `online_golden`
- pin online baseline only after promotion criteria are met

Typical commands:

```powershell
python scripts\build_frozen_corpus.py --config benchmark\benchmark_v1.yaml
python scripts\run_stage5a_gate.py
python scripts\run_stage5b_gate.py
python scripts\run_stage6_preflight.py
python scripts\run_stage4_stop_materialization_revalidation.py
python scripts\run_stage6_shadow_gate.py
python scripts\run_stage6_online_shadow_smoke.py
python scripts\run_stage6_online_shadow_stability.py --repeats 3
python scripts\run_stage6_mpc_tuning.py
python scripts\run_system_benchmark.py --config benchmark\benchmark_v1.yaml --set smoke
python scripts\run_system_benchmark.py --config benchmark\benchmark_v1.yaml --set golden
python scripts\run_system_benchmark.py --config benchmark\benchmark_v1.yaml --set failure_replay
python scripts\run_system_benchmark.py --config benchmark\benchmark_v1.yaml --set golden --compare-baseline benchmark\baselines\system_benchmark_v1\baseline_metrics.json
python scripts\run_system_benchmark_e2e.py --config benchmark\benchmark_v1.yaml --set scenario_replay_smoke --benchmark-mode scenario_replay --execute-stages
```

Stage 6 MPC tuning outputs:
- `benchmark/stage6_mpc_tuning_baseline.json`
- `benchmark/stage6_mpc_tuning_parameter_space.json`
- `benchmark/stage6_mpc_tuning_plan.json`
- `benchmark/stage6_mpc_tuning_runs.json`
- `benchmark/stage6_mpc_tuning_comparison.json`
- `benchmark/stage6_mpc_tuning_decision.json`
- optional selected-default override: `benchmark/stage6_mpc_shadow_default_config.json`

Stage 6 six-case shadow expansion:
- `python scripts\run_stage6_shadow_expansion.py --repeats 5 --carla-root D:/carla`
- outputs:
  - `benchmark/stage6_shadow_expansion_audit.json`
  - `benchmark/stage6_shadow_expansion_report.json`
  - `benchmark/stage6_shadow_expansion_result.json`
  - `benchmark/stage6_shadow_stability_6case_report.json`
  - `benchmark/stage6_shadow_stability_6case_result.json`
  - fixed set: `benchmark/sets/stage6_shadow_golden_6case.yaml`
- current status:
  - smoke is `pass`
  - golden is `pass`
  - stability is `ready`
  - `benchmark/stage6_shadow_stability_6case_result.json` is the readiness authority for the six-case ring
- runtime note:
  - the six-case online shadow ring restarts CARLA between online-e2e cases to isolate simulator state across independent benchmark cases
  - use this path when validating six-case stability; do not reuse a single CARLA world across the full ring
- backend note:
  - the current shadow MPC backend includes a near-zero stop-distance fallback profile to avoid false solver infeasibility when stop target distance collapses to `0.0 m` at a non-zero vehicle speed
  - this is a backend robustness fix, not a benchmark threshold change

Frozen corpus outputs:
- `benchmark/frozen_corpus/v1/manifest.json`
- `benchmark/frozen_corpus/v1/frozen_corpus_audit.json`
- `benchmark/frozen_corpus/v1/frozen_corpus_build_report.json`
- `benchmark/frozen_corpus/v1/frozen_corpus_readiness.json`
- `benchmark/mapping_benchmark_to_frozen_corpus.json`

v1.1 cleanup / repin utilities:

```powershell
python scripts\benchmark_pending_audit.py --config benchmark\benchmark_v1.yaml --output benchmark\reports\benchmark_pending_audit.json
python scripts\ground_truth_backfill.py --config benchmark\benchmark_v1.yaml --output benchmark\reports\ground_truth_backfill_plan.json --apply
python scripts\baseline_repin.py --config benchmark\benchmark_v1.yaml --decision-output benchmark\reports\baseline_repin_decision.json --apply
python scripts\benchmark_readiness_gate.py --config benchmark\benchmark_v1.yaml --stage5a-gate-result benchmark\stage5a_gate_result.json --output benchmark\reports\benchmark_readiness_gate.json
```

- full six-case bring-up without cross-case CARLA contamination:

```powershell
python scripts\run_stage6_shadow_matrix.py --retry-attempts 1 --carla-root D:/carla
```

- repeated six-case bring-up campaign:

```powershell
python scripts\run_stage6_shadow_matrix_campaign.py --repeats 5 --retry-attempts 1 --carla-root D:/carla
```

- pre-takeover sandbox gate:

```powershell
python scripts\run_stage6_pre_takeover_sandbox_gate.py
python scripts\run_stage6_fallback_authority_ring.py
python scripts\run_stage6_control_authority_handoff_sandbox.py
python scripts\run_stage6_control_authority_bounded_motion_sandbox.py
python scripts\run_stage6_control_authority_transfer_sandbox.py
python scripts\run_stage6_organic_fallback_ring.py
python scripts\run_stage6_takeover_sandbox_gate.py
python scripts\run_stage6_multi_case_takeover_sandbox.py
python scripts\run_stage6_takeover_ring_sandbox.py
python scripts\run_stage6_takeover_ring_stability.py --repeats 3
python scripts\run_stage6_takeover_ring_6case_sandbox.py
python scripts\run_stage6_takeover_ring_6case_stability.py --repeats 3
python scripts\run_stage6_takeover_enable_gate.py
python scripts\run_stage6_takeover_rollout_design.py
python scripts\run_stage6_takeover_canary.py --phase phase0
python scripts\run_stage6_takeover_canary.py --phase phase1
```

- pre-takeover sandbox outputs:
  - `benchmark/stage6_control_authority_sandbox_audit.json`
  - `benchmark/stage6_fallback_authority_ring_report.json`
  - `benchmark/stage6_fallback_authority_ring_result.json`
  - `benchmark/stage6_control_authority_handoff_sandbox_report.json`
  - `benchmark/stage6_control_authority_handoff_sandbox_result.json`
  - `benchmark/stage6_control_authority_bounded_motion_sandbox_report.json`
  - `benchmark/stage6_control_authority_bounded_motion_sandbox_result.json`
  - `benchmark/stage6_control_authority_transfer_sandbox_report.json`
  - `benchmark/stage6_control_authority_transfer_sandbox_result.json`
  - `benchmark/stage6_organic_fallback_ring_report.json`
  - `benchmark/stage6_organic_fallback_ring_result.json`
  - `benchmark/stage6_pre_takeover_sandbox_report.json`
  - `benchmark/stage6_pre_takeover_sandbox_result.json`
  - `benchmark/stage6_takeover_sandbox_gate_report.json`
  - `benchmark/stage6_takeover_sandbox_gate_result.json`
  - `benchmark/stage6_multi_case_takeover_sandbox_report.json`
  - `benchmark/stage6_multi_case_takeover_sandbox_result.json`
  - `benchmark/stage6_takeover_ring_sandbox_report.json`
  - `benchmark/stage6_takeover_ring_sandbox_result.json`
  - `benchmark/stage6_takeover_ring_stability_report.json`
  - `benchmark/stage6_takeover_ring_stability_result.json`
  - `benchmark/stage6_takeover_ring_6case_sandbox_report.json`
  - `benchmark/stage6_takeover_ring_6case_sandbox_result.json`
  - `benchmark/stage6_takeover_ring_6case_stability_report.json`
  - `benchmark/stage6_takeover_ring_6case_stability_result.json`
  - `benchmark/stage6_takeover_enable_gate_report.json`
  - `benchmark/stage6_takeover_enable_gate_result.json`
  - `benchmark/stage6_takeover_rollout_design.json`
  - `benchmark/stage6_takeover_rollout_checklist.json`
  - `benchmark/stage6_takeover_rollout_result.json`
  - `benchmark/stage6_takeover_canary_phase0_report.json`
  - `benchmark/stage6_takeover_canary_phase0_result.json`
- interpretation:
  - `sandbox_gate_status = pass` means the promoted ring is clean enough to start explicit takeover sandbox design work
  - `takeover_discussion_status = ready` means the promoted six-case shadow ring, multi-case fallback fault-injection ring, no-motion handoff sandbox, bounded-motion handoff sandbox, and bounded-motion shadow authority transfer sandbox are all clean enough to start explicit takeover sandbox design work
  - `takeover_enable_status = not_ready` is intentional:
    - this gate is a pre-takeover readiness gate, not a takeover-enable gate
    - `benchmark/stage6_organic_fallback_ring_result.json` is now `ready` and `benchmark/stage6_pre_takeover_sandbox_result.json` currently reports `remaining_blockers = []`
    - control authority must still stay on the baseline until a separate takeover-specific gate is defined and passed
- takeover-specific sandbox gate:
  - `benchmark/stage6_takeover_sandbox_gate_result.json` is the authority for the first bounded authority-transfer sandbox gate
  - current truth:
    - `takeover_sandbox_gate_status = pass`
    - `next_scope_status = ready_for_multi_case_takeover_sandbox_design`
    - `takeover_enable_status = not_ready`
  - what it checks:
    - pre-takeover gate dependency is already clean
    - shadow actuator authority is actually applied in the bounded transfer sandbox
    - handoff duration is explicitly bounded
    - rollback back to baseline is observed within one tick after the handoff window
    - no over-budget or hard-health failures occur during the authority window
    - estimated handoff distance stays within the bounded-motion envelope
  - this gate still does not authorize takeover rollout; it only clears the next step of multi-case takeover sandbox design
- narrow multi-case takeover sandbox gate:
  - `benchmark/stage6_multi_case_takeover_sandbox_result.json` is the authority for the first two-case takeover sandbox ring
  - current truth:
    - `overall_status = pass`
    - `next_scope_status = ready_for_takeover_ring_sandbox_expansion`
    - `takeover_enable_status = not_ready`
  - current scope:
    - nominal case: `ml_right_positive_core`
    - stop/arbitration case: `arbitration_stop_during_prepare_right`
  - added expectations:
    - bounded shadow authority transfer must pass on both cases
    - the stop/arbitration case must also keep:
      - `stop_alignment_ready = true`
      - `stop_alignment_consistent = true`
      - `stop_target_exact = true`
      - `stop_final_error_supported = true`
  - this still does not authorize takeover rollout; it only clears the next step of broadening bounded takeover sandbox coverage beyond a single case
- expanded takeover ring sandbox gate:
  - `benchmark/stage6_takeover_ring_sandbox_result.json` is the authority for the first four-case bounded takeover sandbox ring
  - current truth:
    - `overall_status = pass`
    - `next_scope_status = ready_for_takeover_ring_stability_campaign`
    - `takeover_enable_status = not_ready`
  - current scope:
    - nominal cases:
      - `ml_right_positive_core`
      - `ml_left_positive_core`
    - stop-focused cases:
      - `stop_follow_ambiguity_core`
      - `arbitration_stop_during_prepare_right`
  - gate expectations:
    - bounded shadow authority transfer must pass on all four cases
    - stop-focused cases must also keep:
      - `stop_alignment_ready = true`
      - `stop_alignment_consistent = true`
      - `stop_target_exact = true`
      - `stop_final_error_supported = true`
  - this still does not authorize takeover rollout; it only clears the next step of repeatability testing on the expanded takeover sandbox ring
- takeover ring stability campaign:
  - `benchmark/stage6_takeover_ring_stability_result.json` is the authority for repeatability of the four-case bounded takeover sandbox ring
  - current truth:
    - `overall_status = ready`
    - `num_runs = 3`
    - `pass_rate = 1.0`
    - `runtime_success_rate = 1.0`
    - `remaining_blockers = []`
  - aggregate stability:
    - `handoff_duration_ticks_observed` stays pinned at `3`
    - `rollback_delay_ticks` stays pinned at `1`
    - `handoff_distance_estimate_m p95` remains well below the bounded-distance threshold
  - this still does not authorize takeover rollout; it only clears the next step of designing a wider takeover sandbox ring
- six-case takeover ring sandbox gate:
  - `benchmark/stage6_takeover_ring_6case_sandbox_result.json` is the authority for expanding bounded takeover coverage to the promoted six-case ring
  - current truth:
    - `overall_status = pass`
    - `next_scope_status = ready_for_takeover_ring_6case_stability_campaign`
    - `takeover_enable_status = not_ready`
  - current scope:
    - nominal cases:
      - `ml_right_positive_core`
      - `ml_left_positive_core`
      - `junction_straight_core`
      - `fr_lc_commit_no_permission`
    - stop-focused cases:
      - `stop_follow_ambiguity_core`
      - `arbitration_stop_during_prepare_right`
  - this still does not authorize takeover rollout; it only clears repeatability testing on the bounded six-case takeover ring
- six-case takeover ring stability campaign:
  - `benchmark/stage6_takeover_ring_6case_stability_result.json` is the authority for repeatability of the bounded six-case takeover sandbox ring
  - current truth:
    - `overall_status = ready`
    - `num_runs = 3`
    - `pass_rate = 1.0`
    - `runtime_success_rate = 1.0`
    - `remaining_blockers = []`
  - this still does not authorize takeover rollout; it only clears the next step of stricter takeover-enable gate design
- takeover-enable gate:
  - `benchmark/stage6_takeover_enable_gate_result.json` is the authority for consolidated enablement readiness on the bounded six-case takeover ring
  - current truth:
    - `takeover_enable_status = not_ready`
  - this gate still does not authorize takeover rollout by itself; it only decides whether explicit rollout design work can start without unresolved sandbox blockers
- takeover rollout design:
  - `benchmark/stage6_takeover_rollout_result.json` is the authority for the current explicit rollout design package
  - current truth:
    - `takeover_enable_status = not_ready`
  - this package defines rollout phases, kill-switches, abort criteria, and operator checklist items
  - it still does not execute or authorize takeover rollout by itself
- phase0 takeover canary:
  - `benchmark/stage6_takeover_canary_phase0_result.json` is the authority for the first explicit bounded takeover canary on `ml_left_positive_core`
  - this canary still does not authorize broader takeover rollout; it only clears the next stop-focused canary review if it passes
- phase1 stop canary:
  - `benchmark/stage6_takeover_canary_phase1_result.json` is the authority for the first explicit stop-focused bounded takeover canary on `stop_follow_ambiguity_core`
  - this canary still does not authorize broader takeover rollout; it only clears the next nominal ring-extension review if it passes
- phase2 nominal ring extension:
  - `benchmark/stage6_takeover_phase2_result.json` is the authority for expanding explicit bounded takeover to:
    - `ml_right_positive_core`
    - `junction_straight_core`
    - `fr_lc_commit_no_permission`
  - current truth:
    - `overall_status = pass`
    - `next_scope_status = ready_for_phase3_full_ring_candidate_review`
    - `takeover_enable_status = not_ready`
  - this phase still does not authorize broader takeover rollout; it only clears the final promoted-ring stop/arbitration review if it passes
- phase3 full-ring candidate:
  - `benchmark/stage6_takeover_phase3_result.json` is the authority for the final explicit bounded takeover phase on `arbitration_stop_during_prepare_right`
  - current truth:
    - `overall_status = pass`
    - `next_scope_status = ready_for_mpc_e2e_completion_gate_review`
    - `takeover_enable_status = not_ready`
  - this phase still does not authorize broader takeover rollout; it only clears the final end-to-end completion review
- MPC end-to-end completion gate:
  - `benchmark/stage6_mpc_e2e_completion_result.json` is the authority for distinguishing bounded takeover completion from full end-to-end MPC completion
  - current truth:
    - `bounded_takeover_status = complete`
    - `mpc_end_to_end_status = complete`
    - `next_scope_status = complete`
  - current blocker set is explicit:
    - none
  - straight conclusion:
    - the bounded takeover path is complete
    - the repo now has full-scenario MPC authority completion evidence on the promoted six-case ring
- full-authority MPC canaries:
  - `benchmark/stage6_full_authority_canary_phase0_result.json` is the authority for the first nominal full-authority MPC canary on `ml_left_positive_core`
  - `benchmark/stage6_full_authority_canary_phase1_result.json` is the authority for the first stop-focused full-authority MPC canary on `stop_follow_ambiguity_core`
  - current truth:
    - both canaries are `pass`
    - both keep `full_authority_until_end_observed = true`
    - the stop canary also keeps `stop_alignment_ready = true`, `stop_target_exact = true`, and `stop_final_error_supported = true`
- full-authority MPC promoted ring:
  - `benchmark/stage6_full_authority_ring_result.json` is the authority for single-pass full-authority execution on the promoted six-case ring
  - current truth:
    - `overall_status = pass`
    - fallback under full authority is observed on:
      - `ml_right_positive_core`
      - `fr_lc_commit_no_permission`
  - `benchmark/stage6_full_authority_ring_stability_result.json` is the authority for repeatability of full-authority execution on the promoted six-case ring
  - current truth:
    - `overall_status = ready`
    - `pass_rate = 1.0`
    - `runtime_success_rate = 1.0`
    - all six cases pass without retry recovery
  - straight conclusion:
    - promoted-ring full-authority execution exists and fallback under authority has been observed
    - full-authority repeatability is now clean enough to clear the last end-to-end MPC completion blocker

- matrix runner notes:
  - runs one case per CARLA lifecycle
  - retries infra failures once
  - waits for CARLA process exit plus memory drain before the next case
  - waits for CARLA RPC readiness plus runtime stabilization before handing control to Stage 4
  - uses low-memory Stage 4 shadow profiles for the six-case bring-up path
  - keeps baseline as the only control authority
  - use the matrix path as the default way to validate the full 6-case ring when CARLA stability matters more than throughput
  - latest 5-run matrix campaign is clean: `benchmark/stage6_shadow_matrix_campaign_result.json` reports `overall_status = pass` and `pass_rate = 1.0`
  - `benchmark/stage6_shadow_stability_6case_result.json` is now synced from the matrix campaign and should be treated as the readiness authority for the promoted 6-case ring
