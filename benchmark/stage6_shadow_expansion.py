"""Shim: ``benchmark.stage6_shadow_expansion`` → ``agent_ai.benchmark.stage6_shadow_expansion``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.benchmark.stage6_shadow_expansion")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
