"""Shim: ``stage3c_coverage.junction_execution_runner`` → ``agent_ai.behavior.coverage.junction_execution_runner``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.behavior.coverage.junction_execution_runner")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
