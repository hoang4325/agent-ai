"""Shim: ``stage9.tor_manager`` → ``agent_ai.authority.tor_manager``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.authority.tor_manager")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
