"""Shim: ``stage3b.replay_runner_stage3b`` → ``agent_ai.behavior.route.replay_runner_stage3b``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.behavior.route.replay_runner_stage3b")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
