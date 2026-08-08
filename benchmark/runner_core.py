"""Shim: ``benchmark.runner_core`` → ``agent_ai.benchmark.runner_core``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.benchmark.runner_core")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
