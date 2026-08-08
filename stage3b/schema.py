"""Shim: ``stage3b.schema`` → ``agent_ai.behavior.route.schema``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.behavior.route.schema")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
