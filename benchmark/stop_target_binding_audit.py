from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .io import dump_json, load_json, load_yaml, resolve_repo_path


STOP_CASE_HINTS = ("stop", "yield")


def _is_stop_case(case_spec: dict[str, Any]) -> bool:
    expected = [str(item) for item in (case_spec.get("expected_behaviors") or [])]
    forbidden = [str(item) for item in (case_spec.get("forbidden_behaviors") or [])]
    case_id = str(case_spec.get("scenario_id") or "")
    if any(any(hint in value for hint in STOP_CASE_HINTS) for value in expected + forbidden):
        return True
    return "stop" in case_id


def _best_stop_target_path(
    *,
    case_id: str,
    report_roots: Iterable[Path],
    frozen_bundle_dir: Path | None,
) -> tuple[Path | None, str]:
    preferred_relatives = [
        (Path("case_runs") / case_id / "stage4" / "stop_target.json", "online_current_run"),
        (Path("case_runs") / case_id / "stage3c" / "stop_target.json", "scenario_replay_current_run"),
    ]
    for relative, scope in preferred_relatives:
        for report_root in report_roots:
            candidate = report_root / relative
            if candidate.exists():
                return candidate, scope
    if frozen_bundle_dir is not None:
        candidate = frozen_bundle_dir / "reference_outputs" / "stage3c" / "stop_target.json"
        if candidate.exists():
            return candidate, "frozen_reference"
    return None, "missing"


def _classify_stop_target(payload: dict[str, Any]) -> tuple[str, str | None, list[str]]:
    stop_events = list(payload.get("stop_events") or [])
    if not stop_events:
        return "missing", None, ["no_stop_events"]
    binding_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    notes: list[str] = []
    for row in stop_events:
        binding_counter[str(row.get("binding_status") or "unavailable")] += 1
        source_type = str(row.get("target_source_type") or "unavailable")
        if source_type and source_type != "unavailable":
            source_counter[source_type] += 1
    measurement_support = dict(payload.get("measurement_support") or {})
    if str(measurement_support.get("stop_final_error_m")) == "missing":
        notes.append("stop_final_error_m_missing")
    if binding_counter.get("exact", 0) == len(stop_events):
        return "exact", (source_counter.most_common(1)[0][0] if source_counter else None), notes
    if binding_counter.get("approximate", 0) == 0 and binding_counter.get("unavailable", 0) == 0 and (
        binding_counter.get("heuristic", 0) + binding_counter.get("exact", 0) == len(stop_events)
    ):
        return "heuristic", (source_counter.most_common(1)[0][0] if source_counter else None), notes
    return "approximate", (source_counter.most_common(1)[0][0] if source_counter else None), notes


def run_stop_target_binding_audit(
    repo_root: str | Path,
    *,
    report_roots: Iterable[str | Path] | None = None,
    case_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_config = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    mapping = load_json(
        resolve_repo_path(
            repo_root,
            ((benchmark_config.get("frozen_corpus") or {}).get("mapping_path")) or "benchmark/mapping_benchmark_to_frozen_corpus.json",
        ),
        default={},
    ) or {}
    cases_dir = resolve_repo_path(repo_root, benchmark_config.get("cases_dir")) or (repo_root / "benchmark" / "cases")
    resolved_report_roots = [Path(path).resolve() for path in (report_roots or [])]
    if not resolved_report_roots:
        latest_online = sorted(repo_root.glob("benchmark/reports/e2e_online_e2e_*"))
        latest_replay = sorted(repo_root.glob("benchmark/reports/e2e_scenario_replay_*"))
        resolved_report_roots.extend([path.resolve() for path in [*(latest_online[-1:] or []), *(latest_replay[-1:] or [])]])

    case_rows: list[dict[str, Any]] = []
    total_events = 0
    defined_events = 0
    exact_events = 0
    heuristic_events = 0
    missing_cases: list[str] = []

    selected_case_ids = {str(case_id) for case_id in (case_ids or [])}

    for case_path in sorted(cases_dir.glob("*.yaml")):
        case_spec = load_yaml(case_path)
        case_id = str(case_spec.get("scenario_id") or case_path.stem)
        if selected_case_ids and case_id not in selected_case_ids:
            continue
        if not _is_stop_case(case_spec):
            continue
        frozen_bundle_dir = None
        mapping_case = ((mapping.get("cases") or {}).get(case_id)) or {}
        bundle_dir = mapping_case.get("bundle_dir")
        if bundle_dir:
            frozen_bundle_dir = resolve_repo_path(repo_root, bundle_dir)
        artifact_path, source_scope = _best_stop_target_path(
            case_id=case_id,
            report_roots=resolved_report_roots,
            frozen_bundle_dir=frozen_bundle_dir,
        )
        payload = load_json(artifact_path, default={}) if artifact_path is not None else {}
        stop_events = list(payload.get("stop_events") or [])
        total_events += len(stop_events)
        defined_events += sum(1 for row in stop_events if bool(row.get("target_defined")))
        exact_events += sum(1 for row in stop_events if str(row.get("binding_status")) == "exact")
        heuristic_events += sum(1 for row in stop_events if str(row.get("binding_status")) == "heuristic")
        stop_target_status, source_type, notes = _classify_stop_target(payload)
        if stop_target_status == "missing":
            missing_cases.append(case_id)
        case_rows.append(
            {
                "case_id": case_id,
                "stop_target_status": stop_target_status,
                "target_source_type": source_type,
                "source_scope": source_scope,
                "artifact_path": None if artifact_path is None else str(artifact_path),
                "num_stop_events": len(stop_events),
                "defined_events": sum(1 for row in stop_events if bool(row.get("target_defined"))),
                "binding_counts": dict(Counter(str(row.get("binding_status") or "unavailable") for row in stop_events)),
                "notes": notes,
            }
        )

    summary = {
        "defined_rate": (None if total_events == 0 else float(defined_events / total_events)),
        "exact_rate": (None if total_events == 0 else float(exact_events / total_events)),
        "heuristic_rate": (None if total_events == 0 else float(heuristic_events / total_events)),
        "missing_cases": sorted(missing_cases),
        "audited_case_count": len(case_rows),
        "total_stop_events": int(total_events),
    }
    payload = {
        "schema_version": "stop_target_binding_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "cases": case_rows,
    }
    dump_json(repo_root / "benchmark" / "stop_target_binding_audit.json", payload)
    dump_json(repo_root / "benchmark" / "stop_target_binding_summary.json", summary)
    return {"audit": payload, "summary": summary}
