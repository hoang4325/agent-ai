"""Shim: ``stage2.evaluation`` → ``agent_ai.world_state.evaluation``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.world_state.evaluation")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
