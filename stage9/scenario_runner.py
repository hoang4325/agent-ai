"""Shim: ``stage9.scenario_runner`` → ``agent_ai.authority.scenario_runner``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.authority.scenario_runner")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
