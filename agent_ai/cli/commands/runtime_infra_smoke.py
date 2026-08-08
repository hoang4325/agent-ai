from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from agent_ai.cli.bootstrap import REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()
PROJECT_ROOT = REPO_ROOT
from agent_ai.shared.artifact_io import load_json as _load_json  # noqa: E402
from agent_ai.shared.artifact_io import write_json as _write_json  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage4 infrastructure smoke checks.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--carla-pythonapi-root", default=r"D:\carla\PythonAPI")
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--carla-timeout-s", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_path = output_dir / "stage4_runtime_audit.json"
    audit_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "stage4_runtime_audit.py").resolve()),
        "--output",
        str(audit_path),
        "--carla-pythonapi-root",
        str(args.carla_pythonapi_root),
        "--carla-host",
        str(args.carla_host),
        "--carla-port",
        str(args.carla_port),
        "--carla-timeout-s",
        str(args.carla_timeout_s),
    ]

    proc = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    log_path = output_dir / "stage4_infra_smoke.log"
    log_path.write_text(
        "COMMAND: " + " ".join(command) + "\n"
        + f"EXIT_CODE: {proc.returncode}\n\n"
        + (proc.stdout or "")
        + ("\n[STDERR]\n" + proc.stderr if proc.stderr else ""),
        encoding="utf-8",
    )

    audit: Dict[str, Any] = {}
    if audit_path.exists():
        audit = _load_json(audit_path)

    summary = {
        "smoke_version": "stage4_infra_smoke_v1",
        "started_at_utc": _utc_now(),
        "status": "pass" if audit.get("status") == "pass" else "fail",
        "checks": {
            "runtime_audit_invocation_exit_code": int(proc.returncode),
            "runtime_audit_status": audit.get("status"),
            "carla_import_ok": bool(audit.get("import_check", {}).get("ok", False)),
            "carla_connect_ok": bool(audit.get("simulator_connect_check", {}).get("ok", False)),
            "matching_python_dist": bool(
                audit.get("compatibility", {}).get("has_matching_dist_for_active_python", False)
            ),
        },
        "artifacts": {
            "runtime_audit": str(audit_path),
            "smoke_log": str(log_path),
        },
    }
    _write_json(output_dir / "stage4_infra_smoke_summary.json", summary)
    print(
        "[stage4-infra-smoke] "
        f"status={summary['status']} "
        f"import_ok={summary['checks']['carla_import_ok']} "
        f"connect_ok={summary['checks']['carla_connect_ok']}"
    )
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
