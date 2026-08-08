"""Shim: ``benchmark.stage8_temporal_memory`` → ``agent_ai.benchmark.stage8_temporal_memory``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.benchmark.stage8_temporal_memory")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
