"""Shim: ``stage4.world_behavior_runtime`` → ``agent_ai.runtime.world_behavior_runtime``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.runtime.world_behavior_runtime")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
