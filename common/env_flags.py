"""Shim: ``common.env_flags`` → ``agent_ai.shared.env_flags``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.shared.env_flags")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
