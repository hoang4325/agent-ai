"""Shim: ``stage2.risk_engine`` → ``agent_ai.world_state.risk_engine``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.world_state.risk_engine")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
