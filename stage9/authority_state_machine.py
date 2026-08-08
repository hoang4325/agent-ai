"""Shim: ``stage9.authority_state_machine`` → ``agent_ai.authority.authority_state_machine``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.authority.authority_state_machine")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
