"""Shim: ``carla_bevfusion_stage1.coordinate_utils`` → ``agent_ai.perception.coordinate_utils``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.perception.coordinate_utils")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
