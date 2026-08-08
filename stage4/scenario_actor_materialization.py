"""Shim: ``stage4.scenario_actor_materialization`` → ``agent_ai.runtime.scenario_actor_materialization``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.runtime.scenario_actor_materialization")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
