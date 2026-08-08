"""Shim: ``benchmark.carla_runtime`` → ``agent_ai.benchmark.carla_runtime``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.benchmark.carla_runtime")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
