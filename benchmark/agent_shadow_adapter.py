"""Shim: ``benchmark.agent_shadow_adapter`` → ``agent_ai.benchmark.agent_shadow_adapter``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.benchmark.agent_shadow_adapter")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
