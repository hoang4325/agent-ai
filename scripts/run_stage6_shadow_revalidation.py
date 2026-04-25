from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.stage6_shadow_revalidation import run_stage6_shadow_revalidation


if __name__ == "__main__":
    run_stage6_shadow_revalidation(REPO_ROOT)
