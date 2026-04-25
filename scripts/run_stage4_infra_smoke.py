from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


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
        audit = json.loads(audit_path.read_text(encoding="utf-8"))

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
