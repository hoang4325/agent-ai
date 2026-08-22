from __future__ import annotations

import unittest

from scripts.run_stage10_paired_ab_campaign import (
    _blocker_distance_args,
    _geometry_signature,
)


class PairedABManifestTests(unittest.TestCase):
    def test_blocker_distance_override_is_optional(self) -> None:
        self.assertEqual(_blocker_distance_args(None), [])

    def test_blocker_distance_override_is_forwarded(self) -> None:
        self.assertEqual(
            _blocker_distance_args(16.0),
            ["--blocker-distance-m", "16.0"],
        )

    def test_blocker_distance_override_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _blocker_distance_args(0.0)

    def test_actor_ids_do_not_affect_geometry_signature(self) -> None:
        base = {
            "town": "Carla/Maps/Town10HD_Opt",
            "random_seed": 2,
            "adjacent_side": "right",
            "blocker_kind": "vehicle",
            "ego_actor_id": 10,
            "blocker_actor_id": 11,
            "corridor": {
                "road_id": 1,
                "section_id": 0,
                "lane_id": -1,
                "s": 2.0,
                "adjacent_lane_id": -2,
                "ego_transform": {"location": {"x": 1.0, "y": 2.0, "z": 0.1}},
            },
            "placements": {"blocker_distance_m": 10.0},
        }
        assist = dict(base)
        assist["ego_actor_id"] = 20
        assist["blocker_actor_id"] = 21
        self.assertEqual(_geometry_signature(base), _geometry_signature(assist))


if __name__ == "__main__":
    unittest.main()
