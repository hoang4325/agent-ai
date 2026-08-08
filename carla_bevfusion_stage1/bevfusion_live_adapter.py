"""Shim: ``carla_bevfusion_stage1.bevfusion_live_adapter`` → ``agent_ai.perception.bevfusion_live_adapter``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.perception.bevfusion_live_adapter")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
