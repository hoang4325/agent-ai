"""Shim: ``stage3c_coverage.evaluation_stage3c_coverage`` → ``agent_ai.behavior.coverage.evaluation_stage3c_coverage``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.behavior.coverage.evaluation_stage3c_coverage")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
