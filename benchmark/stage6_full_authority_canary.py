"""Shim: ``benchmark.stage6_full_authority_canary`` → ``agent_ai.benchmark.stage6_full_authority_canary``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.benchmark.stage6_full_authority_canary")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
