"""Shim: ``stage3c.local_planner_bridge`` → ``agent_ai.behavior.execution.local_planner_bridge``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.behavior.execution.local_planner_bridge")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
