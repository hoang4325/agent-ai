from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from stage3c.local_planner_bridge import ensure_carla_pythonapi

CARLA_PROCESS_NAMES: tuple[str, ...] = (
    "CarlaUE4-Win64-Shipping",
    "CarlaUE4",
    "CrashReportClient",
)


def port_ready(host: str, port: int, *, timeout_seconds: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(float(timeout_seconds))
        try:
            sock.connect((host, int(port)))
        except OSError:
            return False
    return True


def rpc_ready(
    *,
    carla_root: Path,
    port: int,
    timeout_seconds: float = 10.0,
) -> bool:
    try:
        ensure_carla_pythonapi(carla_root / "PythonAPI")
        import carla  # type: ignore

        client = carla.Client("127.0.0.1", int(port))
        client.set_timeout(float(timeout_seconds))
        world = client.get_world()
        _ = world.get_map().name
        return True
    except Exception:
        return False


def wait_for_carla_runtime_stable(
    *,
    carla_root: Path,
    port: int,
    timeout_s: float = 30.0,
    consecutive_successes: int = 3,
    poll_seconds: float = 1.0,
) -> dict[str, Any]:
    ensure_carla_pythonapi(carla_root / "PythonAPI")
    import carla  # type: ignore

    deadline = time.time() + float(timeout_s)
    stable_count = 0
    last_error: str | None = None
    observations: list[dict[str, Any]] = []
    while time.time() < deadline:
        try:
            client = carla.Client("127.0.0.1", int(port))
            client.set_timeout(10.0)
            world = client.get_world()
            snapshot = world.get_snapshot()
            actor_count = len(world.get_actors())
            observations.append(
                {
                    "frame": int(snapshot.frame),
                    "elapsed_seconds": float(snapshot.timestamp.elapsed_seconds),
                    "actor_count": int(actor_count),
                }
            )
            stable_count += 1
            if stable_count >= max(1, int(consecutive_successes)):
                return {
                    "status": "stable",
                    "observations": observations[-int(consecutive_successes) :],
                }
        except Exception as exc:
            last_error = str(exc)
            stable_count = 0
        time.sleep(float(poll_seconds))
    return {
        "status": "timeout",
        "last_error": last_error,
        "observations": observations,
    }


def ensure_carla_ready(
    *,
    carla_root: Path,
    port: int,
    timeout_s: float = 240.0,
    offscreen: bool = True,
) -> dict[str, Any]:
    if port_ready("127.0.0.1", int(port)) and rpc_ready(carla_root=carla_root, port=int(port)):
        stability = wait_for_carla_runtime_stable(carla_root=carla_root, port=int(port))
        if stability.get("status") != "stable":
            raise RuntimeError(
                "CARLA RPC became reachable but runtime did not stabilize. "
                f"details={stability}"
            )
        return {
            "status": "ready",
            "launched": False,
            "port": int(port),
            "carla_root": str(carla_root),
            "rpc_ready": True,
            "runtime_stability": stability,
        }

    exe_path = carla_root / "CarlaUE4.exe"
    if not exe_path.exists():
        raise FileNotFoundError(f"CARLA executable not found at {exe_path}")

    command = [str(exe_path)]
    if offscreen:
        command.extend(["-RenderOffScreen", "-nosound"])
    command.extend([f"-carla-rpc-port={int(port)}", "-quality-level=Low"])
    subprocess.Popen(command, cwd=str(carla_root))

    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if port_ready("127.0.0.1", int(port)) and rpc_ready(carla_root=carla_root, port=int(port)):
            stability = wait_for_carla_runtime_stable(carla_root=carla_root, port=int(port))
            if stability.get("status") != "stable":
                raise RuntimeError(
                    "CARLA RPC became reachable but runtime did not stabilize. "
                    f"details={stability}"
                )
            return {
                "status": "ready",
                "launched": True,
                "port": int(port),
                "carla_root": str(carla_root),
                "rpc_ready": True,
                "runtime_stability": stability,
            }
        time.sleep(2.0)
    raise RuntimeError(f"CARLA port {port} did not become ready within {timeout_s:.0f}s.")


def carla_process_snapshot() -> list[dict[str, Any]]:
    process_names = ",".join(f"'{name}'" for name in CARLA_PROCESS_NAMES)
    command = (
        f"Get-Process -Name {process_names} -ErrorAction SilentlyContinue "
        "| Select-Object ProcessName,Id,WS "
        "| ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = (result.stdout or "").strip()
    if not payload:
        return []
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if isinstance(decoded, dict):
        decoded = [decoded]
    snapshot: list[dict[str, Any]] = []
    for row in decoded or []:
        snapshot.append(
            {
                "name": str(row.get("ProcessName") or ""),
                "pid": int(row.get("Id") or 0),
                "working_set_bytes": int(row.get("WS") or 0),
            }
        )
    return snapshot


def wait_for_carla_process_exit(
    *,
    timeout_s: float = 90.0,
    poll_seconds: float = 2.0,
    memory_drain_seconds: float = 10.0,
) -> dict[str, Any]:
    deadline = time.time() + float(timeout_s)
    last_snapshot = carla_process_snapshot()
    while time.time() < deadline:
        snapshot = carla_process_snapshot()
        if not snapshot:
            time.sleep(float(memory_drain_seconds))
            return {
                "status": "exited",
                "memory_drain_seconds": float(memory_drain_seconds),
                "last_snapshot": last_snapshot,
            }
        last_snapshot = snapshot
        time.sleep(float(poll_seconds))
    return {
        "status": "timeout",
        "memory_drain_seconds": 0.0,
        "last_snapshot": last_snapshot,
    }


def kill_carla_processes(*, sleep_seconds: float = 3.0) -> None:
    for process_name in tuple(f"{name}.exe" for name in CARLA_PROCESS_NAMES):
        subprocess.run(
            ["taskkill", "/IM", process_name, "/F", "/T"],
            capture_output=True,
            text=True,
            check=False,
        )
    time.sleep(float(sleep_seconds))


def restart_carla(*, carla_root: Path, port: int, timeout_s: float = 240.0) -> dict[str, Any]:
    kill_carla_processes()
    port_close_deadline = time.time() + 30.0
    while port_ready("127.0.0.1", int(port)) and time.time() < port_close_deadline:
        time.sleep(2.0)
    process_release = wait_for_carla_process_exit()
    if process_release.get("status") != "exited":
        raise RuntimeError(
            "CARLA processes did not fully exit before restart. "
            f"last_snapshot={process_release.get('last_snapshot')}"
        )
    return ensure_carla_ready(carla_root=carla_root, port=int(port), timeout_s=float(timeout_s))


def restart_carla_with_retry(
    *,
    carla_root: Path,
    port: int,
    timeout_s: float = 300.0,
    attempts: int = 3,
    backoff_seconds: float = 15.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt_index in range(1, max(1, int(attempts)) + 1):
        try:
            return restart_carla(carla_root=carla_root, port=int(port), timeout_s=float(timeout_s))
        except Exception as exc:
            last_error = exc
            if attempt_index >= max(1, int(attempts)):
                break
            time.sleep(float(backoff_seconds))
    raise RuntimeError(
        "CARLA restart failed after retry budget. "
        f"attempts={max(1, int(attempts))} last_error={last_error}"
    ) from last_error
