from __future__ import annotations

import sys
from pathlib import Path

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
PROJECT_ROOT = REPO_ROOT
from agent_ai.behavior.lane.replay_runner import main


if __name__ == "__main__":
    main()
