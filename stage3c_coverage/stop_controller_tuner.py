"""Shim: ``stage3c_coverage.stop_controller_tuner`` → ``agent_ai.behavior.coverage.stop_controller_tuner``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.behavior.coverage.stop_controller_tuner")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
