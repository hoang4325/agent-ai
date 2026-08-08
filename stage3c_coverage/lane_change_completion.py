"""Shim: ``stage3c_coverage.lane_change_completion`` → ``agent_ai.behavior.coverage.lane_change_completion``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.behavior.coverage.lane_change_completion")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
