"""
Soft-constraint maneuver arbiter (P1).

Hard constraints filter infeasible maneuvers.
Soft constraints contribute additive costs; lowest cost among feasible wins.

This is intentionally pure / side-effect free so tactical policy, maneuver
validation, and route staging can share the same ranking logic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence


@dataclass
class ManeuverCandidate:
    maneuver: str
    hard_ok: bool
    hard_reason: str | None = None
    soft_cost: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Default soft weights (tunable).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "safety": 3.0,
    "gap": 1.5,
    "progress": 1.0,
    "comfort": 0.8,
    "route": 1.2,
    "risk": 2.0,
    "hysteresis": 0.6,
    "preference": 0.5,
}


class SoftConstraintArbiter:
    """
    Rank / select maneuvers under hard + soft constraints.

    Usage:
        arb = SoftConstraintArbiter()
        ranked = arb.rank(candidates)
        best = arb.select(candidates)
    """

    def __init__(self, *, weights: Mapping[str, float] | None = None) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({str(k): float(v) for k, v in weights.items()})

    def total_cost(self, components: Mapping[str, float]) -> float:
        total = 0.0
        for key, value in components.items():
            w = float(self.weights.get(key, 1.0))
            total += w * float(value)
        return float(total)

    def finalize(self, candidate: ManeuverCandidate) -> ManeuverCandidate:
        """Recompute soft_cost from components using current weights."""
        candidate.soft_cost = self.total_cost(candidate.components)
        return candidate

    def rank(self, candidates: Iterable[ManeuverCandidate]) -> List[ManeuverCandidate]:
        items = [self.finalize(c) for c in candidates]
        feasible = [c for c in items if c.hard_ok]
        infeasible = [c for c in items if not c.hard_ok]
        feasible.sort(key=lambda c: (c.soft_cost, c.maneuver))
        infeasible.sort(key=lambda c: (c.soft_cost, c.maneuver))
        return feasible + infeasible

    def select(
        self,
        candidates: Iterable[ManeuverCandidate],
        *,
        fallback: str = "keep_lane",
    ) -> ManeuverCandidate:
        ranked = self.rank(candidates)
        for c in ranked:
            if c.hard_ok:
                return c
        # Nothing feasible — synthesize a keep_lane fallback if present, else first.
        for c in ranked:
            if c.maneuver == fallback:
                return c
        if ranked:
            return ranked[0]
        return ManeuverCandidate(
            maneuver=fallback,
            hard_ok=True,
            soft_cost=0.0,
            tags=["arbiter_empty_fallback"],
        )

    def explain(self, candidate: ManeuverCandidate) -> List[str]:
        tags = list(candidate.tags)
        if not candidate.hard_ok:
            tags.append(f"hard_fail:{candidate.hard_reason or 'unknown'}")
        ordered = sorted(candidate.components.items(), key=lambda kv: -abs(kv[1]))
        for key, value in ordered[:4]:
            tags.append(f"cost_{key}={value:.2f}")
        tags.append(f"total={candidate.soft_cost:.2f}")
        return tags


def build_soft_components(
    *,
    safety: float = 0.0,
    gap: float = 0.0,
    progress: float = 0.0,
    comfort: float = 0.0,
    route: float = 0.0,
    risk: float = 0.0,
    hysteresis: float = 0.0,
    preference: float = 0.0,
) -> Dict[str, float]:
    return {
        "safety": float(safety),
        "gap": float(gap),
        "progress": float(progress),
        "comfort": float(comfort),
        "route": float(route),
        "risk": float(risk),
        "hysteresis": float(hysteresis),
        "preference": float(preference),
    }


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def inverse_gap_cost(gap_m: float | None, *, comfortable_m: float, critical_m: float) -> float:
    """1.0 when gap <= critical, 0.0 when gap >= comfortable, linear in between."""
    if gap_m is None:
        return 0.0  # open road
    if gap_m <= critical_m:
        return 1.0
    if gap_m >= comfortable_m:
        return 0.0
    return clamp01(1.0 - (gap_m - critical_m) / max(1e-3, comfortable_m - critical_m))


def ttc_cost(ttc_s: float | None, *, critical_s: float = 1.5, safe_s: float = 5.0) -> float:
    if ttc_s is None:
        return 0.0
    if ttc_s <= critical_s:
        return 1.0
    if ttc_s >= safe_s:
        return 0.0
    return clamp01(1.0 - (ttc_s - critical_s) / max(1e-3, safe_s - critical_s))
