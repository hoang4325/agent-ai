from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_stage10_stress_tables import _load_rows, _paired_ab_statistics


class Stage10StatisticsTests(unittest.TestCase):
    def test_assist_query_count_is_loaded_without_compare_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "blocked_lane_clear_right_assist_seed0_run1"
            run_dir.mkdir()
            (run_dir / "stage10_driving_metrics.json").write_text(
                json.dumps(
                    {
                        "random_seed": 0,
                        "frames": 10,
                        "collision_count": 0,
                        "lane_invasion_count": 0,
                        "offroad_rate": 0.0,
                        "route": {"route_progress_m": 5.0, "distance_traveled_m": 5.0},
                        "runtime": {"control_loop_latency": {"p95_ms": 80.0}},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "stage10_agent_assist_evaluation.json").write_text(
                json.dumps(
                    {
                        "random_seed": 0,
                        "sim_frames": 10,
                        "agent_query_frames": 2,
                        "agent_query_rate": 0.2,
                        "assist_applied_frames": 3,
                        "latency": {"p95_api_call_ms": 900.0},
                    }
                ),
                encoding="utf-8",
            )
            rows = _load_rows(Path(tmp), "*")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["agent_queried_frames"], 2)
            self.assertEqual(rows[0]["query_ratio"], 0.2)
            self.assertEqual(rows[0]["control_loop_p95_ms"], 80.0)

    def test_paired_difference_uses_common_seed(self) -> None:
        rows = [
            {"case": "case_baseline", "random_seed": 0, "route_progress_m": 2.0},
            {"case": "case_assist", "random_seed": 0, "route_progress_m": 5.0},
        ]
        for row in rows:
            for field in (
                "route_completion_rate",
                "distance_traveled_m",
                "collision_count",
                "lane_invasion_count",
                "offroad_rate",
                "mean_abs_longitudinal_jerk_mps3",
                "max_abs_longitudinal_jerk_mps3",
                "maneuver_duration_s",
                "agreement_rate",
                "agent_fallback_rate",
                "assist_query_rejection_rate",
                "assist_p95_api_call_ms",
                "assist_over_step_budget_rate",
                "stale_response_discard_rate",
                "control_loop_p95_ms",
                "control_loop_over_budget_rate",
                "lane_change_completion_time_s",
            ):
                row[field] = None
        paired = _paired_ab_statistics(rows)
        self.assertEqual(paired["case"]["num_pairs"], 1)
        self.assertEqual(paired["case"]["metrics"]["route_progress_m"]["mean"], 3.0)


if __name__ == "__main__":
    unittest.main()
