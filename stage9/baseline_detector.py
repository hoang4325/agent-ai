"""Shim: ``stage9.baseline_detector`` → ``agent_ai.authority.baseline_detector``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.authority.baseline_detector")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
