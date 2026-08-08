"""Shim: ``stage3c.evaluation_stage3c`` → ``agent_ai.behavior.execution.evaluation_stage3c``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.behavior.execution.evaluation_stage3c")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
