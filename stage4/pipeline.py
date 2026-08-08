"""Shim: ``stage4.pipeline`` → ``agent_ai.runtime.pipeline``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.runtime.pipeline")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
