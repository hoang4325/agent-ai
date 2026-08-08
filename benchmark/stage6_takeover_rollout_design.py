"""Shim: ``benchmark.stage6_takeover_rollout_design`` → ``agent_ai.benchmark.stage6_takeover_rollout_design``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.benchmark.stage6_takeover_rollout_design")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
