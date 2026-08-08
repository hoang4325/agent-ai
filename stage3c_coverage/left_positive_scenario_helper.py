"""Shim: ``stage3c_coverage.left_positive_scenario_helper`` → ``agent_ai.behavior.coverage.left_positive_scenario_helper``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.behavior.coverage.left_positive_scenario_helper")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
