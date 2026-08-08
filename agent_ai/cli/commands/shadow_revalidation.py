from __future__ import annotations

import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
from agent_ai.benchmark.shadow.shadow_revalidation import run_shadow_revalidation


if __name__ == "__main__":
    run_shadow_revalidation(REPO_ROOT)
