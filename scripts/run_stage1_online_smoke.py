from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List


def _host_to_container(host_path: Path, project_root: Path, container_agent_root: str) -> str:
    relative = host_path.resolve().relative_to(project_root.resolve())
    return str(PurePosixPath(container_agent_root) / PurePosixPath(*relative.parts))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _ensure_container_running(container_name: str, cwd: Path) -> None:
    inspect = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode != 0:
        raise RuntimeError(f"docker inspect failed for {container_name}: {(inspect.stderr or '').strip()}")
    if (inspect.stdout or "").strip() == "running":
        return
    start = subprocess.run(
        ["docker", "start", container_name],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if start.returncode != 0:
        raise RuntimeError(f"docker start failed for {container_name}: {(start.stderr or '').strip()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Stage1 online watch smoke.")
    parser.add_argument("--project-root", default=r"D:\Agent-AI")
    parser.add_argument("--samples-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--session-name", default="stage1_online_smoke")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--idle-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.25)
    parser.add_argument("--container-name", default="bevfusion-cu128")
    parser.add_argument("--container-agent-root", default="/workspace/Agent-AI")
    parser.add_argument("--container-repo-root", default="/workspace/Bevfusion-Z")
    parser.add_argument(
        "--container-config",
        default="/workspace/Bevfusion-Z/configs/nuscenes/det/transfusion/secfpn/camera+lidar+radar/swint_v0p075/sefuser_radarbev.yaml",
    )
    parser.add_argument("--container-checkpoint", default="/workspace/Data-Train/epoch_8_3sensor.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summary-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    samples_root = Path(args.samples_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    _ensure_container_running(args.container_name, project_root)

    container_samples_root = _host_to_container(samples_root, project_root, args.container_agent_root)
    container_output_root = _host_to_container(output_root, project_root, args.container_agent_root)

    python_parts = [
        "python -u",
        shlex.quote(f"{args.container_agent_root}/scripts/live_infer.py"),
        "--samples-root",
        shlex.quote(container_samples_root),
        "--repo-root",
        shlex.quote(args.container_repo_root),
        "--config",
        shlex.quote(args.container_config),
        "--checkpoint",
        shlex.quote(args.container_checkpoint),
        "--output-root",
        shlex.quote(container_output_root),
        "--session-name",
        shlex.quote(args.session_name),
        "--num-samples",
        shlex.quote(str(args.num_samples)),
        "--device",
        shlex.quote(args.device),
        "--adapter-lidar-sweeps-test",
        "3",
        "--adapter-radar-sweeps",
        "3",
        "--adapter-lidar-max-points",
        "120000",
        "--adapter-radar-max-points",
        "1200",
        "--radar-bridge",
        "minimal",
        "--radar-ablation",
        "zero_bev",
        "--poll-interval-seconds",
        shlex.quote(str(args.poll_interval_seconds)),
        "--idle-timeout-seconds",
        shlex.quote(str(args.idle_timeout_seconds)),
        "--no-compare-zero-radar",
    ]
    bash_command = " ".join(["source /root/miniconda3/bin/activate bevfusion", "&&", *python_parts])
    command = ["docker", "exec", args.container_name, "/bin/bash", "-lc", bash_command]

    proc = subprocess.run(
        command,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )

    smoke_log_path = output_root / "stage1_online_smoke.log"
    smoke_log_path.write_text(
        "COMMAND: " + " ".join(command) + "\n"
        + f"EXIT_CODE: {proc.returncode}\n\n"
        + (proc.stdout or "")
        + ("\n[STDERR]\n" + proc.stderr if proc.stderr else ""),
        encoding="utf-8",
    )

    session_root = output_root / args.session_name
    summary_rows = _read_jsonl(session_root / "live_summary.jsonl")
    statuses = [str(item.get("status")) for item in summary_rows]
    inferred = sum(1 for s in statuses if s in {"inferred", "inferred_compare"})
    inferred_fail = sum(1 for s in statuses if s.startswith("error"))
    seen_samples = len({str(item.get("sample_name")) for item in summary_rows if item.get("sample_name")})
    stale_events = max(0, seen_samples - inferred)

    result = {
        "num_samples_seen": seen_samples,
        "num_infers_success": inferred,
        "num_infers_fail": inferred_fail,
        "worker_exit_code": int(proc.returncode),
        "smoke_pass": bool(proc.returncode == 0 and inferred >= 3 and inferred_fail == 0),
        "stale_events": stale_events,
        "session_root": str(session_root),
        "smoke_log": str(smoke_log_path),
    }

    summary_output = Path(args.summary_output).resolve() if args.summary_output else (output_root / "stage1_online_smoke_summary.json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        "[stage1-online-smoke] "
        f"pass={result['smoke_pass']} "
        f"seen={result['num_samples_seen']} "
        f"inferred={result['num_infers_success']} "
        f"exit={result['worker_exit_code']}"
    )
    if not result["smoke_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
