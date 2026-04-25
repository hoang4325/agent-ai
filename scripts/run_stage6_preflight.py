"""
run_stage6_preflight.py
=======================
Refreshes frozen corpus, builds the Stage 6 preflight metric pack, audits
stop_target / latency / fallback contracts, and materializes a replay-based
benchmark report that surfaces the new preflight metrics.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.frozen_corpus_builder import build_frozen_corpus
from benchmark.io import dump_json, load_yaml, resolve_repo_path
from benchmark.runner_core import run_benchmark
from benchmark.stage6_preflight_contract_audit import run_stage6_preflight_contract_audit
from benchmark.stage6_preflight_metric_pack import build_stage6_preflight_metric_pack


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _materialize_runtime_overlay(repo_root: Path, benchmark_config: dict[str, Any], metric_pack: dict[str, Any]) -> tuple[Path, str]:
    overlay_root = repo_root / "benchmark" / "reports" / f"stage6_preflight_runtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    overlay_cases_dir = overlay_root / "cases"
    overlay_sets_dir = overlay_root / "sets"
    overlay_cases_dir.mkdir(parents=True, exist_ok=True)
    overlay_sets_dir.mkdir(parents=True, exist_ok=True)

    cases_dir = resolve_repo_path(repo_root, benchmark_config["cases_dir"])
    replay_requirement = dict((metric_pack.get("case_requirements") or {}).get("replay_reference_set") or {})
    base_set_name = str(replay_requirement.get("set_name") or "stage5b_core_golden")
    base_set_path = resolve_repo_path(repo_root, (benchmark_config.get("sets") or {}).get(base_set_name))
    base_set = load_yaml(base_set_path)
    case_ids = list(base_set.get("cases") or [])

    for case_id in case_ids:
        case_spec = load_yaml(cases_dir / f"{case_id}.yaml")
        mandatory = list(dict.fromkeys([
            *(case_spec.get("mandatory_artifacts") or []),
            *(replay_requirement.get("mandatory_artifacts_add") or []),
        ]))
        secondary = list(dict.fromkeys([
            *(case_spec.get("secondary_metrics") or []),
            *(replay_requirement.get("secondary_metrics_add") or []),
        ]))
        thresholds = dict(case_spec.get("metric_thresholds") or {})
        for metric in metric_pack.get("hard_gate_metrics") or []:
            threshold = dict(metric.get("threshold") or {})
            if threshold and metric.get("metric") in {"stop_target_defined_rate", "fallback_reason_coverage"}:
                thresholds[str(metric["metric"])] = threshold
        case_spec["mandatory_artifacts"] = mandatory
        case_spec["secondary_metrics"] = secondary
        case_spec["metric_thresholds"] = thresholds
        case_spec["stage6_preflight_overlay"] = {"enabled": True}
        _write_yaml(overlay_cases_dir / f"{case_id}.yaml", case_spec)

    overlay_set_name = "stage6_preflight_replay"
    _write_yaml(
        overlay_sets_dir / f"{overlay_set_name}.yaml",
        {
            "set_name": overlay_set_name,
            "description": "Replay-side preflight report for Stage 6 contracts",
            "cases": case_ids,
        },
    )

    runtime_cfg = dict(benchmark_config)
    runtime_cfg["cases_dir"] = str(overlay_cases_dir)
    runtime_cfg["sets"] = dict(runtime_cfg.get("sets") or {})
    runtime_cfg["sets"][overlay_set_name] = str(overlay_sets_dir / f"{overlay_set_name}.yaml")
    runtime_cfg_path = overlay_root / "benchmark_runtime_config.yaml"
    _write_yaml(runtime_cfg_path, runtime_cfg)
    return runtime_cfg_path, overlay_set_name


def run_stage6_preflight(repo_root: str | Path | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root or REPO_ROOT)
    benchmark_config = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")

    frozen_root = resolve_repo_path(
        repo_root,
        ((benchmark_config.get("frozen_corpus") or {}).get("root")) or "benchmark/frozen_corpus/v1",
    )
    if frozen_root is None:
        raise ValueError("Missing frozen corpus root in benchmark config.")

    frozen_build = build_frozen_corpus(
        repo_root=repo_root,
        config_path="benchmark/benchmark_v1.yaml",
        output_root=frozen_root,
    )
    metric_pack = build_stage6_preflight_metric_pack(repo_root)
    audit = run_stage6_preflight_contract_audit(repo_root)
    runtime_cfg_path, set_name = _materialize_runtime_overlay(repo_root, benchmark_config, metric_pack)
    replay_eval = run_benchmark(
        repo_root=repo_root,
        config_path=runtime_cfg_path,
        set_name=set_name,
        report_dir=repo_root / "benchmark" / "reports" / f"{set_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    payload = {
        "schema_version": "stage6_preflight_result_v1",
        "metric_pack_path": str(repo_root / "benchmark" / "stage6_preflight_metric_pack.json"),
        "contract_audit_path": str(repo_root / "benchmark" / "stage6_preflight_contract_audit.json"),
        "replay_report_dir": str(replay_eval["report_dir"]),
        "frozen_corpus_readiness": (frozen_build.get("readiness") or {}).get("summary"),
        "contract_audit": {
            "stop_target_contract": audit.get("stop_target_contract"),
            "latency_contract": audit.get("latency_contract"),
            "fallback_contract": audit.get("fallback_contract"),
        },
    }
    dump_json(repo_root / "benchmark" / "stage6_preflight_result.json", payload)
    return payload


if __name__ == "__main__":
    run_stage6_preflight()
