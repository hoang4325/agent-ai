from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import dump_json, load_json


def _required_file(path: Path) -> dict[str, Any]:
    return {"path": str(path), "passed": path.exists()}


def validate_case_bundle(bundle_dir: Path) -> dict[str, Any]:
    case_manifest_path = bundle_dir / "case_manifest.json"
    scenario_manifest_path = bundle_dir / "scenario_manifest.json"
    benchmark_reference_path = bundle_dir / "benchmark_reference" / "benchmark_reference.json"
    provenance_path = bundle_dir / "provenance" / "provenance.json"
    case_manifest = load_json(case_manifest_path, default={})

    checks = {
        "case_manifest": _required_file(case_manifest_path),
        "scenario_manifest": _required_file(scenario_manifest_path),
        "benchmark_reference": _required_file(benchmark_reference_path),
        "provenance": _required_file(provenance_path),
        "stage1_session_summary": _required_file(bundle_dir / "stage1_frozen" / "session" / "session_summary.json"),
        "stage1_live_summary": _required_file(bundle_dir / "stage1_frozen" / "session" / "live_summary.jsonl"),
        "stage2_reference": _required_file(bundle_dir / "reference_outputs" / "stage2" / "evaluation_summary.json"),
        "stage3a_reference": _required_file(bundle_dir / "reference_outputs" / "stage3a" / "evaluation_summary.json"),
        "stage3b_reference": _required_file(bundle_dir / "reference_outputs" / "stage3b" / "evaluation_summary.json"),
        "stage3c_reference": _required_file(bundle_dir / "reference_outputs" / "stage3c" / "execution_summary.json"),
        "semantic_ground_truth": _required_file(bundle_dir / "reference_outputs" / "semantic" / "ground_truth_actor_roles.json"),
        "semantic_route_session_binding": _required_file(bundle_dir / "reference_outputs" / "semantic" / "route_session_binding.json"),
    }
    expected_readiness = str(case_manifest.get("corpus_readiness", "corpus_pending"))
    required_check_names = ["case_manifest", "scenario_manifest", "benchmark_reference", "provenance"]
    if expected_readiness == "corpus_ready":
        required_check_names.extend(
            [
                "stage1_session_summary",
                "stage1_live_summary",
                "stage2_reference",
                "stage3a_reference",
                "stage3b_reference",
                "stage3c_reference",
                "semantic_ground_truth",
                "semantic_route_session_binding",
            ]
        )
    passed = all(checks[name]["passed"] for name in required_check_names)
    return {
        "case_id": case_manifest.get("case_id") or bundle_dir.name,
        "bundle_dir": str(bundle_dir),
        "expected_readiness": expected_readiness,
        "required_checks": required_check_names,
        "passed": passed,
        "checks": checks,
    }


def validate_corpus(corpus_root: Path) -> dict[str, Any]:
    cases_dir = corpus_root / "cases"
    case_results = []
    if cases_dir.exists():
        for bundle_dir in sorted(path for path in cases_dir.iterdir() if path.is_dir()):
            case_results.append(validate_case_bundle(bundle_dir))
    return {
        "corpus_root": str(corpus_root),
        "passed": all(item["passed"] for item in case_results),
        "cases": case_results,
    }


def validate_and_write(corpus_root: Path, output_path: Path) -> dict[str, Any]:
    payload = validate_corpus(corpus_root)
    dump_json(output_path, payload)
    return payload
