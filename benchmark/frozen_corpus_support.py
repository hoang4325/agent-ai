"""Shim: ``benchmark.frozen_corpus_support`` → ``agent_ai.benchmark.frozen_corpus_support``."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("agent_ai.benchmark.frozen_corpus_support")
sys.modules[__name__] = _module
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
