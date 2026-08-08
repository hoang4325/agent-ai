"""Shim: ``stage3b.lane_change_staging`` → ``agent_ai.behavior.route.lane_change_staging``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.behavior.route.lane_change_staging")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
