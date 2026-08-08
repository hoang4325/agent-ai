"""Shim: ``stage4.shadow_runtime`` → ``agent_ai.runtime.shadow_runtime``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.runtime.shadow_runtime")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
