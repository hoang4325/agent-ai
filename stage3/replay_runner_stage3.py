"""Shim: ``stage3.replay_runner_stage3`` → ``agent_ai.behavior.lane.replay_runner_stage3``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.behavior.lane.replay_runner_stage3")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
