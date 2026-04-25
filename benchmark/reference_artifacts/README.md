# Semantic Reference Artifacts

These files contain benchmark-owned semantic grounding inputs for v1.1:
- `ground_truth_actor_roles.json`

They are intentionally lightweight:
- they describe expected roles and binding hints
- they do not claim full simulator actor-id truth unless explicitly available
- they let the benchmark derive stronger semantic metrics from existing artifact bundles

Derived artifacts such as:
- `critical_actor_binding.jsonl`
- `behavior_arbitration_events.jsonl`
- `route_session_binding.json`

are generated on-demand under benchmark report directories unless an explicit case artifact path is provided.
