"""Shim: ``stage4.evaluation_stage4`` → ``agent_ai.runtime.evaluation_stage4``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.runtime.evaluation_stage4")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
