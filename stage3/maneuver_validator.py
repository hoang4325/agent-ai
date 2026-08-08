"""Shim: ``stage3.maneuver_validator`` → ``agent_ai.behavior.lane.maneuver_validator``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.behavior.lane.maneuver_validator")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
