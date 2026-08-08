"""Shim: ``stage9.handoff_planner`` → ``agent_ai.authority.handoff_planner``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.authority.handoff_planner")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
