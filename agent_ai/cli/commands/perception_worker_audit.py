from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _tail(path: Path, max_lines: int = 160) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:]) + ("\n" if lines else "")


def _classify(audit_payload: Dict[str, Any], log_tail: str) -> tuple[str, str]:
    exit_code = audit_payload.get("worker_exit_code")
    tail_lower = log_tail.lower()

    if exit_code in (None, 0) and audit_payload.get("failure_class") and audit_payload.get("root_cause"):
        return str(audit_payload["failure_class"]), str(audit_payload["root_cause"])

    if exit_code in (None, 0):
        return "none", "clean_or_running"
    if "container" in tail_lower and "is not running" in tail_lower:
        return "crash_on_boot", "docker_container_not_running"
    if "traceback" in tail_lower and "infer" in tail_lower:
        return "crash_on_infer", "python_exception_in_inference"
    if "traceback" in tail_lower:
        return "crash_on_boot", "python_exception_on_boot"
    if "samples-root" in tail_lower and "no such file" in tail_lower:
        return "path_mapping_error", "samples_root_mapping_invalid"
    if "meta.json" in tail_lower and "not found" in tail_lower:
        return "bad_sample", "missing_sample_meta"
    return "unknown", "nonzero_exit_without_signature"


def build_audit(stage4_output_dir: Path) -> Dict[str, Any]:
    stage4_output_dir = stage4_output_dir.resolve()
    log_path = stage4_output_dir / "stage1_worker.log"
    prior_audit_path = stage4_output_dir / "stage1_worker_audit.json"
    manifest_path = stage4_output_dir / "stage4_manifest.json"
    summary_path = stage4_output_dir / "online_run_summary.json"
    samples_root = stage4_output_dir / "samples"

    prior = _read_json(prior_audit_path)
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    log_tail = _tail(log_path)

    sample_dirs = sorted([p for p in samples_root.glob("sample_*") if p.is_dir()]) if samples_root.exists() else []
    last_sample = None if not sample_dirs else sample_dirs[-1].name

    effective_exit_code = summary.get("stage1_summary", {}).get("worker_exit_code", prior.get("worker_exit_code"))
    merged_for_classification = dict(prior)
    merged_for_classification["worker_exit_code"] = effective_exit_code
    failure_class, root_cause = _classify(merged_for_classification, log_tail)

    result: Dict[str, Any] = {
        "spawn_command": prior.get("spawn_command"),
        "python_executable": prior.get("python_executable"),
        "cwd": prior.get("cwd"),
        "worker_exit_code": effective_exit_code,
        "failure_class": failure_class,
        "last_seen_sample": prior.get("last_seen_sample") or last_sample,
        "last_emitted_prediction": prior.get("last_emitted_prediction"),
        "root_cause": root_cause,
        "notes": [
            f"stage4_output_dir={stage4_output_dir}",
            f"samples_count={len(sample_dirs)}",
            f"inferred_samples={summary.get('stage1_summary', {}).get('inferred_samples')}",
            f"unresolved_captures={summary.get('stage1_summary', {}).get('unresolved_captures')}",
            f"stage1_launch_mode={manifest.get('args', {}).get('stage1_launch_mode')}",
        ],
    }
    return result, log_tail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage1 watch worker behavior from Stage4 output artifacts.")
    parser.add_argument("--stage4-output-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--failure-trace", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage4_output_dir = Path(args.stage4_output_dir).resolve()
    audit_payload, failure_trace = build_audit(stage4_output_dir)

    output_json = Path(args.output_json).resolve() if args.output_json else (stage4_output_dir / "stage1_worker_audit.json")
    failure_trace_path = Path(args.failure_trace).resolve() if args.failure_trace else (stage4_output_dir / "stage1_worker_failure_trace.txt")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    failure_trace_path.parent.mkdir(parents=True, exist_ok=True)
    failure_trace_path.write_text(failure_trace, encoding="utf-8")

    print(
        "[stage1-worker-audit] "
        f"failure_class={audit_payload['failure_class']} "
        f"exit_code={audit_payload['worker_exit_code']} "
        f"output={output_json}"
    )


if __name__ == "__main__":
    main()
