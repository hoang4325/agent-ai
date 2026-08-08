"""Shim: ``stage9.authority_logger`` → ``agent_ai.authority.authority_logger``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.authority.authority_logger")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
