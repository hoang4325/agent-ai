"""Shim: ``stage4.state_store`` → ``agent_ai.runtime.state_store``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.runtime.state_store")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
