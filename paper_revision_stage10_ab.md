# Stage 10 A/B Study — Manuscript Revision and Reviewer Response

> Status: source-grounded revision draft. The repository does not contain the manuscript file or the original CARLA result folders. Numerical fields marked `[RERUN]` must be filled from a new paired multi-seed campaign before resubmission. Do not replace them with values from the previous single run.

## 1. What the implementation actually evaluates

The two cases are `blocked_lane_clear_right` and `blocked_lane_clear_left`. In both cases, the ego vehicle starts behind a static blocker placed 10 m ahead; the adjacent-lane vehicle is placed 60 m ahead, and the route-progress proxy has a 60 m target. The campaign uses Town10HD_Opt by default. A simulation step is 0.1 s; therefore, the previously reported 15 s experiment corresponds to 150 simulation frames, even though the current campaign default is 300 frames.

The baseline is `RealBaselineAdapter`, a deterministic safety-aware rule policy. It requests `stop_before_obstacle` when minimum TTC is below 1.5 s or the forward corridor is not clear, `follow` for TTC below 4.0 s, and `keep_lane` otherwise. Its nominal cruise speed is 8.0 m/s. The resulting request uses a 3.0 s horizon, maximum longitudinal acceleration 2.5 m/s², maximum lateral acceleration 1.5 m/s², jerk limit 3.0 m/s³, and a 0.75 m lateral bound.

In the Agent-assisted condition, the LLM cannot emit steering, throttle, brake, target speed, waypoints, or a trajectory. It returns only a tactical intent, target lane, confidence, and reason tags. A recommendation is applied only after validation and a bounded assist gate. Otherwise, control falls back to the baseline. The present live bridge implements this as a lightweight assist gate in `_agent_assist_allowed`; it must not be described as the complete Stage 8 `ArbitrationEngine` unless the full engine is actually connected to the live path.

## 2. Point-by-point response to the reviewer

### Comment 1 — Single run, 15 s, and no statistical uncertainty

**Response.** We agree. The original two 15 s runs are pilot observations and do not support population-level or general claims. We revised the experimental protocol to use paired A/B runs with the same seed in each baseline/Agent pair. The campaign now records explicit Python, NumPy, CARLA pedestrian, and Traffic Manager seeds. The summarizer reports the number of independent runs, mean, sample standard deviation, standard error, and a two-sided 95% Student-t confidence interval. We will report results over `[RERUN: N, recommended N ≥ 10]` paired seeds and `[RERUN: duration, recommended 60 s or maneuver completion plus a fixed post-maneuver horizon]` per condition. We also narrowed the conclusion: the blocked-clear results demonstrate behavior in two controlled stress cases on Town10HD_Opt, not general autonomous-driving performance.

**Required table format after rerun.** Report each quantity as `mean ± SE [95% CI]`, separately for right and left, plus the paired Agent-minus-baseline difference using matching seeds. At minimum include route progress/completion, collisions, lane invasions, off-road rate, mean and maximum absolute longitudinal jerk, maneuver completion time, query rejection rate, and API latency.

### Comment 2 — Qwen3-32B/Groq API latency in a real-time control loop

**Response.** We agree and added an explicit latency analysis and limitation. The implementation timestamps every API call and now reports mean, p50, p95, maximum latency, and the fraction of calls exceeding the simulation-step budget. With `delta_t = 0.1 s`, the nominal step budget is 100 ms. The current Groq-compatible request is synchronous and the configured adapter timeout is 30 s; therefore, rate limiting to 30 requests/min does not provide a hard real-time guarantee. A delayed request can stall wall-clock execution even though CARLA simulation time is paused in synchronous mode. The safety benefit of the current design is functional fallback: malformed, invalid, failed, or timed-out outputs are rejected and the baseline remains the control authority. This does not make the external API suitable for deployment in a hard real-time vehicle controller.

We revised the system claim accordingly. Qwen3-32B is a sparse tactical advisor evaluated in simulation, not an actuator-level real-time controller. Production deployment would require asynchronous inference, timestamped world-state snapshots, cancellation of stale responses, a strict deadline substantially below the control period, a cached or local fallback policy, and evaluation under injected network delay and packet loss. These mechanisms are future work and are not claimed by the present implementation.

### Comment 3 — 72% rejection in the left-turn/left-change case

**Response.** We agree that a 72% non-application rate is too high to claim stable Agent–baseline coordination. We revised the terminology from “safety-veto rate” to **Agent query rejection rate**, defined as queried frames on which the recommendation was not applied divided by all queried frames. This distinction matters because the live gate can reject a query for several non-safety reasons: the Agent mirrored the baseline, the API fell back, the output was invalid, confidence was below threshold, the lane change had already completed, or the intent was unsupported. Safety-related causes include lack of lane-change permission and low TTC during continuation. The revised results must report the full rejection-reason histogram rather than treating all rejected queries as one class.

For the reported 72%, insert the actual decomposition from `query_rejection_reason_counts` and `agent_fallback_reason_counts`:

> In Blocked-Clear Left, `[RERUN: rejected]/[RERUN: queried]` queries (`[RERUN]%`) were not applied. The causes were `[RERUN: reason=count, ...]`. Thus, `[RERUN]%` were attributable to safety/feasibility gates and `[RERUN]%` to agreement, API fallback, parsing, or confidence filtering. The high overall rate indicates limited decision utility and temporal coordination in this case, even though rejected recommendations did not override baseline control.

Do not describe the 72% as evidence that the safety layer “worked well” without this decomposition. It simultaneously shows fail-safe containment and poor Agent utility.

### Comment 4 — Limited map/scenario coverage and missing driving metrics

**Response.** We agree. We now report lane-invasion events using CARLA's lane-invasion sensor, off-road frame rate using non-projected driving-lane lookup, longitudinal jerk from finite differences of ego acceleration, total run duration, and lane-change completion time. These supplement route progress/completion, collision count, low-TTC frames, disagreement, fallback, and intervention metrics. We explicitly retain single-map and two-scenario coverage as a limitation. Results from Town10HD_Opt should be described as controlled case-study evidence; evaluation across additional towns, weather, traffic density, occlusion patterns, and dynamic adjacent-lane actors remains necessary for generalization.

### Comment 5 — Baseline, prompt/schema, and parameter reproducibility

**Response.** We expanded the implementation details as follows.

- Baseline policy: TTC/corridor rule policy described above, followed by an OSQP-backed kinematic MPC when available and a proportional fallback otherwise.
- Agent input: baseline intent; preferred lane; route option and conflict flags; risk level; front free space; lane-change permission for each side; and active maneuver.
- Agent output schema: `{"tactical_intent":"...","target_lane":"...","confidence":0.0,"reason_tags":["..."]}`.
- Allowed intents: `keep_lane`, `follow`, `slow_down`, `stop`, `yield`, `prepare_lane_change_left/right`, `commit_lane_change_left/right`, and `keep_route_through_junction`.
- Generation configuration for the OpenAI-compatible endpoint: temperature 0.0, JSON-object mode when supported, maximum 96 tokens for the compact blocked-clear prompt and 150 for the rich prompt, with Qwen reasoning hidden and `/no_think` appended.
- Assist threshold: minimum Agent confidence 0.50. Lane-change assist uses a 3.0 s horizon, speed cap at most 8.0 m/s, longitudinal acceleration limit 2.0 m/s², lateral acceleration limit 1.5 m/s², jerk limit 3.0 m/s³, and bounded target speeds of 0.8–1.5 m/s while preparing and 1.0–2.5 m/s when committing; the active request is subsequently retuned conservatively.

The symbols `λ1, λ2, λ3, w1, w2, w3` do not exist in the current implementation. They should not remain in the manuscript as unexplained implementation parameters. Either remove those conceptual symbols or explicitly map them to the actual MPC coefficients. The source uses the following coefficients:

| Objective | Actual implementation coefficients |
|---|---|
| Longitudinal follow | `q_speed=0.9`, terminal speed `2.2`, acceleration `r=0.18`, acceleration-difference `r=0.3` |
| Longitudinal stop | speed `0.45`, position `0.12`, terminal speed `5.5`, terminal position `18.0`, acceleration `0.18`, acceleration-difference `0.3` |
| Lateral | offset `1.1`, terminal offset `8.0`, lateral velocity `0.15`, control `0.12`, control-difference `0.25` |

If the equations in the manuscript use a reduced three-weight notation, add an explicit mapping table and explain any omitted terms. Do not invent a one-to-one mapping that is absent from the code.

## 3. Ready-to-paste revised manuscript text

### Experimental protocol

> We conducted a paired A/B evaluation in CARLA Town10HD_Opt using two blocked-lane scenarios: Blocked-Clear Right and Blocked-Clear Left. In each scenario, a static vehicle blocked the ego lane 10 m ahead and the adjacent-lane vehicle was positioned 60 m ahead, yielding a clear adjacent corridor near the ego vehicle. The route-progress target was 60 m. The two conditions were (A) the deterministic safety-aware baseline and (B) the same baseline augmented by a Qwen3-32B tactical recommendation queried through the Groq API. Each A/B pair used the same random seed and initial scenario configuration. We used `[RERUN: N]` distinct seeds, a simulation step of 0.1 s, and `[RERUN]` frames per run. We report the mean, sample standard deviation, standard error, and two-sided 95% Student-t confidence interval across seeds. Because the evaluation includes only two scenarios and one map, all conclusions are restricted to these controlled stress cases.

### Baseline and Agent interface

> The baseline maps the perceived world state to one of three tactical behaviors: stop before an obstacle when minimum TTC is below 1.5 s or the forward corridor is blocked, follow when TTC is below 4.0 s, and keep lane otherwise. It generates a bounded trajectory request with an 8.0 m/s nominal cruise speed, 3.0 s horizon, 2.5 m/s² maximum longitudinal acceleration, 1.5 m/s² maximum lateral acceleration, 3.0 m/s³ jerk limit, and 0.75 m lateral bound. The Agent receives a compact semantic state containing the baseline intent, route preference, route-conflict flags, risk level, front free space, left/right lane-change permission, and any active maneuver. It returns JSON containing only a tactical intent, target lane, confidence, and reason tags. Direct actuator and trajectory fields are forbidden. A recommendation is eligible for assist only if it is valid, differs from the baseline, exceeds confidence 0.50, and satisfies the current lane-change permission. Any failed, malformed, stale, low-confidence, or disallowed recommendation falls back to baseline control.

### Latency and real-time limitation

> The external LLM is called synchronously and sparsely under event/risk triggering and a wall-clock rate limit. For every call, we record mean, p50, p95, and maximum API latency and the fraction exceeding the 100 ms simulation-step budget. The measured values were `[RERUN]`. This design provides functional safety fallback but not a hard real-time guarantee: a delayed network request can stall wall-clock execution, and the configured 30 s timeout is far above the control period. Accordingly, we evaluate Qwen3-32B only as a tactical simulation advisor and do not claim deployability as an in-loop real-time vehicle controller. An asynchronous, deadline-aware architecture with stale-response cancellation and a local fallback is required for deployment.

### Metrics and statistical analysis

> Primary outcome metrics were route completion/progress and collision count. We additionally measured lane-invasion count, collision and lane-invasion rates per kilometer, off-road frame rate, low-TTC frames, mean and maximum absolute longitudinal jerk, jerk exceedance rate above 3 m/s³, total maneuver duration, and lane-change completion time. Agent-specific metrics included query rate, application/intervention rate, query rejection rate, fallback rate, agreement with the baseline, rejection reasons, and API latency. Continuous metrics are reported as mean ± standard error with 95% Student-t confidence intervals. Counts and rates are also reported per seed; paired A/B differences are computed using matching seeds.

### Results language that is statistically safe

> Across `[RERUN: N]` paired seeds, Agent assist changed route progress by `[RERUN: mean difference] ± [RERUN: SE] m` in Blocked-Clear Right and `[RERUN] ± [RERUN] m` in Blocked-Clear Left. The corresponding 95% confidence intervals were `[RERUN]` and `[RERUN]`. Collision, lane-invasion, off-road, comfort, and completion-time results are summarized in Table `[RERUN]`. These results indicate `[state only what the confidence intervals support]` for the evaluated configurations; they do not establish general superiority across maps or traffic distributions.

### Limitations

> This study remains limited in scenario diversity and operational scale. It covers two constructed blocked-lane cases on Town10HD_Opt and does not span additional towns, weather conditions, traffic densities, or sensor perturbations. Although multiple seeds quantify within-configuration variability, they do not substitute for broader domain coverage. The Agent uses a remote API whose latency and availability are network-dependent; the present synchronous integration is unsuitable for hard real-time deployment. Finally, a high query rejection rate—particularly in Blocked-Clear Left—shows that Agent outputs and the assist gate are not yet consistently coordinated. The safety fallback prevents rejected recommendations from directly controlling the vehicle, but high rejection reduces utility and motivates better temporal prompting, calibrated confidence, asynchronous inference, and explicit optimization of proposal acceptance subject to unchanged safety constraints.

## 4. Reproduction protocol

Use at least ten paired seeds and the same seed list for both arms. The following is a template; replace paths and the exact model identifier with the values used for the paper.

```bash
export AGENT_API_ENDPOINT="https://api.groq.com/openai/v1/chat/completions"
export AGENT_MODEL_ID="[EXACT QWEN3-32B MODEL ID USED]"
export AGENT_API_KEY="[SET IN YOUR SECRET MANAGER OR SHELL; NEVER COMMIT]"

python scripts/run_stage10_agent_stress_campaign.py \
  --carla-root "[CARLA_ROOT]" \
  --bev-repo "[BEVFUSION_REPO]" \
  --bev-config "[BEVFUSION_CONFIG]" \
  --bev-ckpt "[BEVFUSION_CHECKPOINT]" \
  --cases ab_blocked_clear \
  --seeds 101,202,303,404,505,606,707,808,909,1010 \
  --run-tag baseline \
  --max-frames 600 \
  --delta-t 0.1 \
  --agent-mode stub \
  --agent-control-mode baseline

python scripts/run_stage10_agent_stress_campaign.py \
  --carla-root "[CARLA_ROOT]" \
  --bev-repo "[BEVFUSION_REPO]" \
  --bev-config "[BEVFUSION_CONFIG]" \
  --bev-ckpt "[BEVFUSION_CHECKPOINT]" \
  --cases ab_blocked_clear \
  --seeds 101,202,303,404,505,606,707,808,909,1010 \
  --run-tag assist \
  --max-frames 600 \
  --delta-t 0.1 \
  --agent-mode api \
  --agent-control-mode assist
```

Generate the statistical summary:

```bash
python scripts/summarize_stage10_stress_tables.py \
  --report-root benchmark/reports/stage10_stress_campaign \
  --run-glob 'blocked_lane_clear_*' \
  --summary-json benchmark/reports/stage10_stress_campaign/paper_multiseed_summary.json
```

Before submission, archive the exact environment variables except the API key, CARLA version, BEVFusion commit/config/checkpoint hash, repository commit, GPU/CPU, OS, date/time window of the remote API experiment, seed list, and raw per-run JSON artifacts.

## 5. Claims that must be removed or softened if no rerun is possible

If the multi-seed campaign cannot be completed before resubmission, do not present standard errors from frame-level samples because frames within one run are temporally correlated and are not independent experimental units. Instead:

- Label the results as a two-case, single-run pilot or qualitative case study.
- Remove claims of general superiority, robustness, statistical significance, or real-time readiness.
- Keep the 72% rejection result as a limitation and report its reason breakdown.
- Report observed API latencies descriptively, but do not claim a latency distribution from one or two calls.
- State that multi-seed, multi-map, and network-perturbation evaluation is required future work.
