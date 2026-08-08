"""Shim: ``stage3.lane_context_builder`` → ``agent_ai.behavior.lane.lane_context_builder``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.behavior.lane.lane_context_builder")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
