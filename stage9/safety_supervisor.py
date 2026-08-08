"""Shim: ``stage9.safety_supervisor`` → ``agent_ai.authority.safety_supervisor``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.authority.safety_supervisor")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
