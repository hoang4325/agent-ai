"""Shim: ``benchmark.fallback_activation_probe`` → ``agent_ai.benchmark.fallback_activation_probe``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.benchmark.fallback_activation_probe")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
