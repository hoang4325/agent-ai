"""Shim: ``benchmark.stage7_agent_shadow_audit`` → ``agent_ai.benchmark.stage7_agent_shadow_audit``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.benchmark.stage7_agent_shadow_audit")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
