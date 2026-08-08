"""Shim: ``benchmark.frozen_case_materializer`` → ``agent_ai.benchmark.frozen_case_materializer``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.benchmark.frozen_case_materializer")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
