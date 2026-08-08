"""Shim: ``stage4.online_orchestrator`` → ``agent_ai.runtime.online_orchestrator``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.runtime.online_orchestrator")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
