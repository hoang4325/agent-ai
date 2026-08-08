"""Shim: ``benchmark.stage5b_failure_replay_builder`` → ``agent_ai.benchmark.stage5b_failure_replay_builder``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.benchmark.stage5b_failure_replay_builder")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
