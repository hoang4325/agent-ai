"""Shim: ``stage9.minimal_risk_maneuver`` → ``agent_ai.authority.minimal_risk_maneuver``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.authority.minimal_risk_maneuver")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
