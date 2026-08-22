from __future__ import annotations

import argparse
import unittest

from scripts.run_stage10_stage1_live_bridge import (
    _build_driving_metrics,
    _classify_lane_crossing,
    _group_collision_episodes,
)


class _CollisionMonitor:
    def __init__(self, events):
        self._events = events

    def counted_events(self):
        return list(self._events)

    def counted_episodes(self):
        return _group_collision_episodes(self._events)


class _LaneMonitor:
    def __init__(self, events):
        self._events = events

    def classified_events(self):
        result = []
        for event in self._events:
            classified = dict(event)
            classified["classification"] = _classify_lane_crossing(event)
            result.append(classified)
        return result


class Stage10EventMetricTests(unittest.TestCase):
    def test_consecutive_callbacks_from_same_actor_form_one_collision_episode(self) -> None:
        events = [
            {"frame_id": 100, "timestamp_s": 10.0, "other_actor_id": 7,
             "other_actor_type": "vehicle.test", "intensity": 100.0},
            {"frame_id": 101, "timestamp_s": 10.1, "other_actor_id": 7,
             "other_actor_type": "vehicle.test", "intensity": 5.0},
            {"frame_id": 102, "timestamp_s": 10.2, "other_actor_id": 7,
             "other_actor_type": "vehicle.test", "intensity": 1.0},
        ]
        episodes = _group_collision_episodes(events)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["sensor_event_count"], 3)
        self.assertEqual(episodes[0]["peak_intensity"], 100.0)
        self.assertEqual(episodes[0]["start_frame_id"], 100)
        self.assertEqual(episodes[0]["end_frame_id"], 102)

    def test_separated_or_different_actor_contacts_form_new_episodes(self) -> None:
        events = [
            {"frame_id": 100, "other_actor_id": 7, "other_actor_type": "vehicle.a",
             "intensity": 10.0},
            {"frame_id": 110, "other_actor_id": 7, "other_actor_type": "vehicle.a",
             "intensity": 10.0},
            {"frame_id": 111, "other_actor_id": 8, "other_actor_type": "vehicle.b",
             "intensity": 10.0},
        ]
        self.assertEqual(len(_group_collision_episodes(events)), 3)

    def test_lane_crossing_classification_distinguishes_broken_and_solid(self) -> None:
        legal = {
            "crossed_lane_markings": [
                {"type": "Broken", "color": "White", "lane_change": "Both"}
            ]
        }
        illegal = {
            "crossed_lane_markings": [
                {"type": "Solid", "color": "White", "lane_change": "None"}
            ]
        }
        unknown = {"crossed_lane_markings": []}
        self.assertEqual(_classify_lane_crossing(legal), "legal")
        self.assertEqual(_classify_lane_crossing(illegal), "illegal")
        self.assertEqual(_classify_lane_crossing(unknown), "unknown")

    def test_driving_metrics_publish_episode_and_legality_counts(self) -> None:
        collision_events = [
            {"frame_id": 10, "other_actor_id": 7, "other_actor_type": "vehicle.a",
             "intensity": 20.0},
            {"frame_id": 11, "other_actor_id": 7, "other_actor_type": "vehicle.a",
             "intensity": 2.0},
        ]
        lane_events = [
            {"frame_id": 20, "crossed_lane_markings": [
                {"type": "Broken", "lane_change": "Both"}
            ]},
            {"frame_id": 30, "crossed_lane_markings": [
                {"type": "Solid", "lane_change": "None"}
            ]},
        ]
        args = argparse.Namespace(
            map="Town10HD_Opt",
            seed=0,
            agent_mode="api",
            agent_control_mode="assist",
            delta_t=0.1,
            success_rc_threshold=0.95,
            log_dir="/tmp/stage10-test",
        )
        stats = {
            "frames": 10,
            "errors": 0,
            "longitudinal_jerk_samples_mps3": [],
            "tick_latency_samples_ms": [],
            "lidar_point_samples": [],
            "radar_point_samples": [],
        }
        metrics = _build_driving_metrics(
            args=args,
            stats=stats,
            route_tracker=None,
            collision_monitor=_CollisionMonitor(collision_events),
            lane_invasion_monitor=_LaneMonitor(lane_events),
        )
        self.assertEqual(metrics["collision_count"], 1)
        self.assertEqual(metrics["collision_episode_count"], 1)
        self.assertEqual(metrics["collision_sensor_event_count"], 2)
        self.assertEqual(metrics["lane_invasion_count"], 2)
        self.assertEqual(metrics["legal_lane_crossing_count"], 1)
        self.assertEqual(metrics["illegal_lane_invasion_count"], 1)
        self.assertEqual(metrics["episode_duration_s"], 1.0)
        self.assertFalse(metrics["success_criteria"]["illegal_lane_invasion_passed"])


if __name__ == "__main__":
    unittest.main()
