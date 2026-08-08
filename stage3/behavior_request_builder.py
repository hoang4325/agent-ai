"""Shim: ``stage3.behavior_request_builder`` → ``agent_ai.behavior.lane.behavior_request_builder``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.behavior.lane.behavior_request_builder")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
