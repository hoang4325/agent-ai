# Semantic algorithm cases (B1 / B3)

Lightweight, CARLA-free regression fixtures for world-state / LC / risk behavior.

## Layout

```
semantic_cases/v1/
  cut_in_front_short_ttc.json
  dense_left_gap_ok.json
  rear_fast_block_lc.json
  vru_cross_yield.json
  junction_no_lc.json
```

Each case is a self-contained JSON scene:

- `ego`, `tracks` — synthetic objects in ego frame
- `lane_context` / `lane_objects` — map-relative geometry for LC cost
- `expect` — allowed / forbidden outcomes (`maneuvers_*`, `lc_*`, `risk_*`, `tags_any`, `prediction_modes_any`)

## Run

```bash
# CLI summary (exit 1 if any case fails)
python3 -m agent_ai.benchmark.semantic_case_runner

# Unit regression (CI)
python3 -m unittest tests.test_algorithm_regression -v
```

## Adding a case

1. Copy an existing JSON and set a unique `case_id`.
2. Keep expectations **semantic** (sets of allowed maneuvers), not brittle exact scores.
3. Run the runner; fix either the case or the algorithm if behavior is wrong by design.
