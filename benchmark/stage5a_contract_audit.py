"""Shim: ``benchmark.stage5a_contract_audit`` → ``agent_ai.benchmark.stage5a_contract_audit``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.benchmark.stage5a_contract_audit")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
