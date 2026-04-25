from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.frozen_corpus_builder import build_frozen_corpus  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build frozen_case_bundle_v1 corpus from benchmark artifact bundles.")
    parser.add_argument("--config", default=str(REPO_ROOT / "benchmark" / "benchmark_v1.yaml"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "benchmark" / "frozen_corpus" / "v1"))
    parser.add_argument("--case", dest="cases", action="append", default=None, help="Optional case id to build. Can be repeated.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_frozen_corpus(
        repo_root=REPO_ROOT,
        config_path=args.config,
        output_root=Path(args.output_root),
        selected_cases=args.cases,
    )
    console = {
        "corpus_root": result["manifest"]["corpus_root"],
        "num_cases": result["manifest"]["summary"]["num_cases"],
        "readiness_counts": result["manifest"]["summary"]["readiness_counts"],
        "mapping_path": "benchmark/mapping_benchmark_to_frozen_corpus.json",
    }
    print(json.dumps(console, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
