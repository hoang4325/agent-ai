"""Shim: ``carla_bevfusion_stage1.constants`` → ``agent_ai.perception.constants``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.perception.constants")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
