from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@dataclass
class StepResult:
    name: str
    command: list[str]
    exit_code: int
    elapsed_s: float



def _build_env(*, cpu_threads: int, disable_mp4: bool) -> dict[str, str]:
    env = dict(os.environ)

    thread_cap = str(max(1, int(cpu_threads)))
    env["OMP_NUM_THREADS"] = thread_cap
    env["MKL_NUM_THREADS"] = thread_cap
    env["OPENBLAS_NUM_THREADS"] = thread_cap
    env["NUMEXPR_NUM_THREADS"] = thread_cap
    env.setdefault("MALLOC_ARENA_MAX", "2")

    # Reduce CUDA allocator fragmentation on 6GB VRAM class GPUs.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64,garbage_collection_threshold:0.8")

    if disable_mp4:
        env["AGENTAI_DISABLE_MP4"] = "1"

    return env



def _run_step(
    *,
    name: str,
    command: Sequence[str],
    env: dict[str, str],
    cooldown_s: float,
) -> StepResult:
    started = time.perf_counter()
    proc = subprocess.run(list(command), cwd=str(REPO_ROOT), env=env, check=False)
    elapsed = time.perf_counter() - started

    if cooldown_s > 0.0:
        time.sleep(float(cooldown_s))

    return StepResult(
        name=name,
        command=list(command),
        exit_code=int(proc.returncode),
        elapsed_s=float(elapsed),
    )


def _command_exists(command_name: str) -> bool:
    return shutil.which(command_name) is not None


def _docker_daemon_ready() -> bool:
    if not _command_exists("docker"):
        return False
    proc = subprocess.run(
        ["docker", "version"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and "Server:" in (proc.stdout or "")


def _wsl_has_docker_desktop() -> bool:
    if not _command_exists("wsl"):
        return False
    proc = subprocess.run(
        ["wsl", "-l", "-v"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False
    return "docker-desktop" in (proc.stdout or "")


def _try_start_docker_desktop() -> bool:
    candidates = [
        Path(r"C:/Program Files/Docker/Docker/Docker Desktop.exe"),
        Path(r"C:/Program Files/Docker/Docker/Docker Desktop Installer.exe"),
    ]
    for exe_path in candidates:
        if not exe_path.exists():
            continue
        subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))
        return True
    return False


def _ensure_docker_ready(*, timeout_s: float, poll_s: float = 3.0) -> tuple[bool, str]:
    if _docker_daemon_ready():
        return True, "docker_daemon_already_ready"

    launch_started = _try_start_docker_desktop()
    has_wsl_backend = _wsl_has_docker_desktop()

    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if _docker_daemon_ready():
            if launch_started:
                return True, "docker_daemon_ready_after_autostart"
            return True, "docker_daemon_ready_after_wait"
        time.sleep(float(poll_s))

    reason = "docker_daemon_not_ready"
    if not _command_exists("docker"):
        reason = "docker_cli_missing"
    elif not has_wsl_backend:
        reason = "wsl_docker_desktop_distro_missing"
    elif not launch_started:
        reason = "docker_desktop_executable_not_found"
    return False, reason


def _container_has_live_infer(container_name: str, container_agent_root: str) -> bool:
    proc = subprocess.run(
        [
            "docker",
            "exec",
            str(container_name),
            "test",
            "-f",
            f"{container_agent_root.rstrip('/')}/scripts/live_infer.py",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _resolve_container_agent_root(*, container_name: str) -> str | None:
    candidates: list[str] = []
    env_override = os.environ.get("AGENTAI_STAGE1_CONTAINER_AGENT_ROOT")
    if env_override:
        candidates.append(str(env_override))
    candidates.extend([
        "/workspace/Agent-AI",
        "/workspace/agent-ai",
        "/workspace/AGENT-AI",
        "/workspace/Bevfusion-Z",
    ])

    dedup: list[str] = []
    for item in candidates:
        if item and item not in dedup:
            dedup.append(item)

    for root in dedup:
        if _container_has_live_infer(container_name, root):
            return root
    return None



def _build_steps(*, python_executable: str, mode: str, include_online: bool) -> list[tuple[str, list[str]]]:
    py = str(python_executable)

    smoke_steps: list[tuple[str, list[str]]] = [
        (
            "benchmark_smoke_replay_regression",
            [
                py,
                str(SCRIPTS_DIR / "run_system_benchmark.py"),
                "--set",
                "smoke",
            ],
        ),
        (
            "benchmark_scenario_replay_smoke",
            [
                py,
                str(SCRIPTS_DIR / "run_system_benchmark_e2e.py"),
                "--set",
                "scenario_replay_smoke",
                "--benchmark-mode",
                "scenario_replay",
                "--execute-stages",
            ],
        ),
    ]

    if include_online:
        smoke_steps.append(
            (
                "benchmark_online_e2e_smoke",
                [
                    py,
                    str(SCRIPTS_DIR / "run_system_benchmark_e2e.py"),
                    "--set",
                    "online_e2e_smoke",
                    "--benchmark-mode",
                    "online_e2e",
                    "--execute-stages",
                ],
            )
        )

    if mode == "smoke":
        return smoke_steps

    full_steps = list(smoke_steps)
    full_steps.append(
        (
            "benchmark_golden_replay_regression",
            [
                py,
                str(SCRIPTS_DIR / "run_system_benchmark.py"),
                "--set",
                "golden",
            ],
        )
    )
    return full_steps



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark tests in a low-crash, resource-capped sequence.",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--include-online", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--cooldown-s", type=float, default=3.0)
    parser.add_argument("--disable-mp4", action="store_true", default=True)
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--auto-docker", action="store_true", default=True)
    parser.add_argument("--docker-timeout-s", type=float, default=180.0)
    parser.add_argument("--online-container-name", default="bevfusion-cu128")
    parser.add_argument("--online-stage1-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--summary-path",
        default=str(REPO_ROOT / "outputs" / "test_safe_run_summary.json"),
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()

    env = _build_env(
        cpu_threads=int(args.cpu_threads),
        disable_mp4=bool(args.disable_mp4),
    )
    steps = _build_steps(
        python_executable=str(args.python_executable),
        mode=str(args.mode),
        include_online=bool(args.include_online),
    )

    results: list[StepResult] = []

    print("[safe-test] starting")
    print(f"[safe-test] mode={args.mode} include_online={bool(args.include_online)} cpu_threads={args.cpu_threads}")
    print(f"[safe-test] AGENTAI_DISABLE_MP4={env.get('AGENTAI_DISABLE_MP4', '0')}")

    docker_ready = None
    docker_ready_reason = None
    resolved_container_agent_root = None
    online_skipped_reason = None
    if bool(args.include_online) and bool(args.auto_docker):
        print("[safe-test] checking docker/wsl readiness for online steps")
        docker_ready, docker_ready_reason = _ensure_docker_ready(timeout_s=float(args.docker_timeout_s))
        print(f"[safe-test] docker_ready={docker_ready} reason={docker_ready_reason}")

        if not docker_ready:
            # Skip online step to avoid guaranteed failure; keep replay/golden running.
            steps = [
                (name, cmd)
                for name, cmd in steps
                if name != "benchmark_online_e2e_smoke"
            ]
            online_skipped_reason = "docker_not_ready"
            print("[safe-test] online step skipped because docker is not ready")
        else:
            resolved_container_agent_root = _resolve_container_agent_root(
                container_name=str(args.online_container_name)
            )
            if resolved_container_agent_root:
                env["AGENTAI_STAGE1_CONTAINER_AGENT_ROOT"] = str(resolved_container_agent_root)
                env["AGENTAI_STAGE1_DEVICE"] = str(args.online_stage1_device)
                print(
                    "[safe-test] resolved AGENTAI_STAGE1_CONTAINER_AGENT_ROOT="
                    f"{resolved_container_agent_root}"
                )
                print(f"[safe-test] AGENTAI_STAGE1_DEVICE={env['AGENTAI_STAGE1_DEVICE']}")
            else:
                steps = [
                    (name, cmd)
                    for name, cmd in steps
                    if name != "benchmark_online_e2e_smoke"
                ]
                online_skipped_reason = "live_infer_missing_in_container_agent_root"
                print(
                    "[safe-test] online step skipped because container cannot find "
                    "scripts/live_infer.py under known agent roots"
                )

    for step_name, command in steps:
        print(f"[safe-test] run: {step_name}")
        result = _run_step(
            name=step_name,
            command=command,
            env=env,
            cooldown_s=float(args.cooldown_s),
        )
        results.append(result)
        print(
            f"[safe-test] done: {step_name} exit={result.exit_code} elapsed_s={result.elapsed_s:.2f}"
        )

        if args.stop_on_fail and result.exit_code != 0:
            break

    failed_steps = [item.name for item in results if item.exit_code != 0]
    status = "pass" if not failed_steps else "partial_fail"

    summary = {
        "status": status,
        "mode": str(args.mode),
        "include_online": bool(args.include_online),
        "auto_docker": bool(args.auto_docker),
        "docker_ready": docker_ready,
        "docker_ready_reason": docker_ready_reason,
        "online_skipped_reason": online_skipped_reason,
        "online_container_name": str(args.online_container_name),
        "online_stage1_device": str(args.online_stage1_device),
        "resolved_container_agent_root": resolved_container_agent_root,
        "cpu_threads": int(args.cpu_threads),
        "disable_mp4": bool(args.disable_mp4),
        "failed_steps": failed_steps,
        "steps": [
            {
                "name": item.name,
                "exit_code": item.exit_code,
                "elapsed_s": item.elapsed_s,
                "command": item.command,
            }
            for item in results
        ],
    }

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[safe-test] summary: {summary_path}")
    return 0 if not failed_steps else 1


if __name__ == "__main__":
    raise SystemExit(main())
