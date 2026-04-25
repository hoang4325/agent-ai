from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage3c.local_planner_bridge import ensure_carla_pythonapi


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _cpython_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _discover_dist_artifacts(pythonapi_root: Path) -> List[Dict[str, Any]]:
    dist_dir = pythonapi_root / "carla" / "dist"
    if not dist_dir.exists():
        return []
    rows: List[Dict[str, Any]] = []
    tag_pattern = re.compile(r"(cp\d{2,3})", re.IGNORECASE)
    for item in sorted(dist_dir.iterdir()):
        if not item.is_file():
            continue
        tag_match = tag_pattern.search(item.name)
        rows.append(
            {
                "name": item.name,
                "path": str(item),
                "suffix": item.suffix,
                "detected_cpython_tag": None if tag_match is None else tag_match.group(1).lower(),
            }
        )
    return rows


def build_runtime_audit(args: argparse.Namespace) -> Dict[str, Any]:
    pythonapi_root = Path(args.carla_pythonapi_root).resolve()
    before_path = list(sys.path)

    payload: Dict[str, Any] = {
        "audit_version": "stage4_runtime_audit_v1",
        "status": "fail",
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
            },
            "cpython_tag": _cpython_tag(),
            "platform": platform.platform(),
        },
        "environment": {
            "cwd": str(Path.cwd()),
            "carla_pythonapi_root": str(pythonapi_root),
            "carla_pythonapi_root_exists": pythonapi_root.exists(),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "carla_pythonapi_root_env": os.environ.get("CARLA_PYTHONAPI_ROOT"),
        },
        "carla_pythonapi_dist": _discover_dist_artifacts(pythonapi_root),
        "import_check": {
            "ok": False,
            "error": None,
            "traceback": None,
            "carla_file": None,
            "carla_version": None,
        },
        "simulator_connect_check": {
            "ok": False,
            "error": None,
            "traceback": None,
            "host": str(args.carla_host),
            "port": int(args.carla_port),
            "timeout_s": float(args.carla_timeout_s),
            "world_map": None,
        },
        "path_bootstrap": {
            "added_sys_path_entries": [],
        },
        "compatibility": {
            "has_matching_dist_for_active_python": False,
            "matching_dist_names": [],
        },
    }

    bootstrap_error: str | None = None
    bootstrap_traceback: str | None = None
    try:
        ensure_carla_pythonapi(pythonapi_root)
    except Exception as exc:  # pragma: no cover - diagnostic path
        bootstrap_error = f"{type(exc).__name__}: {exc}"
        bootstrap_traceback = traceback.format_exc()

    after_path = list(sys.path)
    payload["path_bootstrap"]["added_sys_path_entries"] = [p for p in after_path if p not in before_path]
    payload["path_bootstrap"]["error"] = bootstrap_error
    payload["path_bootstrap"]["traceback"] = bootstrap_traceback

    expected_tag = _cpython_tag()
    matching = [
        row["name"]
        for row in payload["carla_pythonapi_dist"]
        if row.get("detected_cpython_tag") == expected_tag
    ]
    payload["compatibility"]["has_matching_dist_for_active_python"] = bool(matching)
    payload["compatibility"]["matching_dist_names"] = matching

    if bootstrap_error is None:
        try:
            import carla  # type: ignore

            has_client = hasattr(carla, "Client")
            payload["import_check"]["ok"] = bool(has_client)
            payload["import_check"]["carla_file"] = getattr(carla, "__file__", None)
            payload["import_check"]["carla_version"] = getattr(carla, "__version__", None)
            if not has_client:
                payload["import_check"]["error"] = "Imported module 'carla' is missing runtime API attribute 'Client'."
        except Exception as exc:  # pragma: no cover - diagnostic path
            payload["import_check"]["error"] = f"{type(exc).__name__}: {exc}"
            payload["import_check"]["traceback"] = traceback.format_exc()
    else:
        payload["import_check"]["error"] = "CARLA bootstrap failed before import check."

    if payload["import_check"]["ok"]:
        import carla  # type: ignore
        try:
            client = carla.Client(str(args.carla_host), int(args.carla_port))
            client.set_timeout(float(args.carla_timeout_s))
            world = client.get_world()
            payload["simulator_connect_check"]["ok"] = True
            payload["simulator_connect_check"]["world_map"] = world.get_map().name
        except Exception as exc:  # pragma: no cover - diagnostic path
            payload["simulator_connect_check"]["error"] = f"{type(exc).__name__}: {exc}"
            payload["simulator_connect_check"]["traceback"] = traceback.format_exc()

    payload["status"] = (
        "pass"
        if payload["import_check"]["ok"] and payload["simulator_connect_check"]["ok"]
        else "fail"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage4 runtime and CARLA import/connectivity readiness.")
    parser.add_argument("--output", required=True, help="Path to write stage4_runtime_audit.json")
    parser.add_argument("--carla-pythonapi-root", default=r"D:\carla\PythonAPI")
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--carla-timeout-s", type=float, default=5.0)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when audit status is fail.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_runtime_audit(args)
    output_path = Path(args.output).resolve()
    _write_json(output_path, payload)
    print(f"[stage4-runtime-audit] status={payload['status']} output={output_path}")
    if args.strict and payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
