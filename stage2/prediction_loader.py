"""Shim: ``stage2.prediction_loader`` → ``agent_ai.world_state.prediction_loader``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.world_state.prediction_loader")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
