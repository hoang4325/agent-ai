from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.fallback_activation_probe import run_fallback_activation_probe


if __name__ == "__main__":
    run_fallback_activation_probe(REPO_ROOT)
