"""Shim: ``stage9.human_override_monitor`` → ``agent_ai.authority.human_override_monitor``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.authority.human_override_monitor")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
