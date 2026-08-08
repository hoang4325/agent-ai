"""Shim: ``stage4.arbitration`` → ``agent_ai.runtime.arbitration``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.runtime.arbitration")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
