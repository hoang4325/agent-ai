"""
Adaptive soft-cost weights with temporal scene memory (P2).

Not a trained ML model — online exponential updates from observed outcomes:
  - near-miss / short TTC → increase safety/risk weights
  - successful stable cruise → slight comfort increase
  - repeated LC abort / hysteresis thrash → increase hysteresis weight
  - route pressure scenes → increase route weight

Also provides scene-profile multipliers (urban / open / junction-like).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

from .soft_constraint_arbiter import DEFAULT_WEIGHTS, SoftConstraintArbiter


@dataclass
class SceneProfile:
    name: str
    multipliers: Dict[str, float]


PROFILES: Dict[str, SceneProfile] = {
    "open": SceneProfile(
        name="open",
        multipliers={"safety": 0.9, "gap": 0.9, "progress": 1.1, "comfort": 1.0, "route": 0.9, "risk": 0.9},
    ),
    "urban": SceneProfile(
        name="urban",
        multipliers={"safety": 1.2, "gap": 1.15, "progress": 0.9, "comfort": 1.05, "route": 1.0, "risk": 1.25},
    ),
    "junction": SceneProfile(
        name="junction",
        multipliers={"safety": 1.35, "gap": 1.2, "progress": 0.75, "comfort": 1.1, "route": 1.4, "risk": 1.3},
    ),
    "dense": SceneProfile(
        name="dense",
        multipliers={"safety": 1.25, "gap": 1.3, "progress": 0.85, "comfort": 1.15, "route": 1.0, "risk": 1.35, "hysteresis": 1.2},
    ),
}


def classify_scene(
    *,
    object_count: int,
    nearest_front_m: float | None,
    highest_risk: str,
    junction_near: bool = False,
    interaction_severity: float = 0.0,
) -> str:
    if junction_near:
        return "junction"
    if object_count >= 8 or interaction_severity > 0.55:
        return "dense"
    if nearest_front_m is not None and nearest_front_m < 18.0:
        return "urban"
    if highest_risk in {"high", "critical"}:
        return "urban"
    return "open"


@dataclass
class CostMemory:
    """
    Temporal memory of soft weights.

    Call `observe(...)` each frame with lightweight outcome signals.
    Call `arbiter_for_scene(...)` to get a SoftConstraintArbiter with adapted weights.
    """

    base_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    adapted: Dict[str, float] = field(default_factory=dict)
    lr: float = 0.08
    decay: float = 0.995
    min_w: float = 0.3
    max_w: float = 5.0
    # rolling stats
    frames: int = 0
    near_miss_count: int = 0
    thrash_count: int = 0
    last_maneuver: str | None = None
    last_profile: str = "open"

    def __post_init__(self) -> None:
        if not self.adapted:
            self.adapted = dict(self.base_weights)

    def _clamp(self, w: float) -> float:
        return float(max(self.min_w, min(self.max_w, w)))

    def _nudge(self, key: str, delta: float) -> None:
        cur = float(self.adapted.get(key, self.base_weights.get(key, 1.0)))
        self.adapted[key] = self._clamp(cur + self.lr * delta)

    def observe(
        self,
        *,
        min_ttc_s: float | None = None,
        highest_risk: str = "low",
        maneuver: str | None = None,
        interaction_severity: float = 0.0,
        object_count: int = 0,
        nearest_front_m: float | None = None,
        junction_near: bool = False,
        lc_aborted: bool = False,
    ) -> None:
        """Update adaptive weights from one observation frame."""
        self.frames += 1
        # Mild decay toward base (prevents runaway).
        for k, base in self.base_weights.items():
            cur = float(self.adapted.get(k, base))
            self.adapted[k] = self._clamp(self.decay * cur + (1.0 - self.decay) * base)

        # Near-miss / short TTC → safety & risk up, progress down.
        if min_ttc_s is not None and min_ttc_s < 2.0:
            self.near_miss_count += 1
            self._nudge("safety", 0.6)
            self._nudge("risk", 0.5)
            self._nudge("progress", -0.3)
        elif min_ttc_s is not None and min_ttc_s < 3.5:
            self._nudge("safety", 0.25)
            self._nudge("risk", 0.2)

        if highest_risk in {"high", "critical"}:
            self._nudge("safety", 0.35)
            self._nudge("gap", 0.2)

        if interaction_severity > 0.4:
            self._nudge("risk", 0.35)
            self._nudge("hysteresis", 0.15)

        # Maneuver thrash: peer switches without safety escalation.
        if (
            maneuver
            and self.last_maneuver
            and maneuver != self.last_maneuver
            and {maneuver, self.last_maneuver} <= {
                "follow",
                "slow_down",
                "lane_change_left",
                "lane_change_right",
                "keep_lane",
            }
        ):
            self.thrash_count += 1
            self._nudge("hysteresis", 0.4)
            self._nudge("comfort", 0.15)
        self.last_maneuver = maneuver

        if lc_aborted:
            self._nudge("gap", 0.35)
            self._nudge("safety", 0.25)
            self._nudge("hysteresis", 0.2)

        # Stable cruise rewards comfort slightly.
        if maneuver in {"keep_lane", "follow"} and (min_ttc_s is None or min_ttc_s > 4.0):
            self._nudge("comfort", 0.05)
            self._nudge("progress", 0.05)

        self.last_profile = classify_scene(
            object_count=object_count,
            nearest_front_m=nearest_front_m,
            highest_risk=highest_risk,
            junction_near=junction_near,
            interaction_severity=interaction_severity,
        )

    def effective_weights(self, profile: str | None = None) -> Dict[str, float]:
        prof_name = profile or self.last_profile
        prof = PROFILES.get(prof_name, PROFILES["open"])
        out: Dict[str, float] = {}
        for k, base in self.base_weights.items():
            adapted = float(self.adapted.get(k, base))
            mult = float(prof.multipliers.get(k, 1.0))
            out[k] = self._clamp(adapted * mult)
        return out

    def arbiter_for_scene(self, profile: str | None = None) -> SoftConstraintArbiter:
        return SoftConstraintArbiter(weights=self.effective_weights(profile))

    def snapshot(self) -> Dict[str, float | int | str]:
        return {
            "frames": self.frames,
            "near_miss_count": self.near_miss_count,
            "thrash_count": self.thrash_count,
            "last_profile": self.last_profile,
            "weights": dict(self.adapted),
            "effective": self.effective_weights(),
        }
