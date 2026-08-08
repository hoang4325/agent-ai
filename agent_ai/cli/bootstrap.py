"""CLI bootstrap helpers — repo root and sys.path setup."""
from __future__ import annotations

import sys
from pathlib import Path

from agent_ai.paths import REPO_ROOT


def ensure_repo_on_path() -> Path:
    """Ensure the repository root is importable, then return it."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


# Convenience alias used by migrated command modules.
PROJECT_ROOT = REPO_ROOT
