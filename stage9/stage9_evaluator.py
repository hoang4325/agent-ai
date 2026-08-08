"""Shim: ``stage9.stage9_evaluator`` → ``agent_ai.authority.stage9_evaluator``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.authority.stage9_evaluator")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
