"""Backward-compatible shim for ``benchmark``.

Canonical package: ``agent_ai.benchmark``.
This shim re-exports the canonical package so legacy imports keep working
during migration. Prefer importing from ``agent_ai.benchmark`` in new code.
"""
from __future__ import annotations

import importlib
import sys
from types import ModuleType

_CANONICAL = "agent_ai.benchmark"


def _load() -> ModuleType:
    module = importlib.import_module(_CANONICAL)
    sys.modules[__name__] = module
    return module


_module = _load()

# Re-export public names for star-import / static analyzers when possible.
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})
__all__ = getattr(_module, "__all__", [k for k in globals() if not k.startswith("_")])
