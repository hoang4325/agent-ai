"""Shared numeric helpers used across stages."""
from __future__ import annotations


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp ``value`` into ``[lower, upper]``."""
    return max(float(lower), min(float(upper), float(value)))


def clamp01(value: float) -> float:
    """Clamp ``value`` into ``[0.0, 1.0]``."""
    return clamp(value, 0.0, 1.0)
