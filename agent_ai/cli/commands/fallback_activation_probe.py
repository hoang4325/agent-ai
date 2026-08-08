from __future__ import annotations

import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.gates.fallback_activation_probe import run_fallback_activation_probe


if __name__ == "__main__":
    run_fallback_activation_probe(REPO_ROOT)
