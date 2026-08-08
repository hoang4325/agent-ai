"""Shim: ``stage2.visualization`` → ``agent_ai.world_state.visualization``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.world_state.visualization")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
