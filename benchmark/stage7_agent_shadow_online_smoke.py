"""Shim: ``benchmark.stage7_agent_shadow_online_smoke`` → ``agent_ai.benchmark.stage7_agent_shadow_online_smoke``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.benchmark.stage7_agent_shadow_online_smoke")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
