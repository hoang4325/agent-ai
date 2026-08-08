"""Shim: ``stage4.control_helpers`` → ``agent_ai.runtime.control_helpers``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.runtime.control_helpers")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
