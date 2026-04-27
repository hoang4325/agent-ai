from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic Stage10 stress scenarios with attached Agent compare."
    )
    parser.add_argument("--carla-root", required=True)
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--bev-repo", required=True)
    parser.add_argument("--bev-config", required=True)
    parser.add_argument("--bev-ckpt", required=True)
    parser.add_argument("--map", default="Town10HD_Opt")
    parser.add_argument("--cases", default="right,left",
                        help="Comma-separated deterministic stress cases to run. Supported: right,left")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--route-distance-m", type=float, default=120.0)
    parser.add_argument("--agent-mode", default="compare", choices=["stub", "api", "compare"])
    parser.add_argument("--agent-trigger-mode", default="risk_or_event",
                        choices=["every_frame", "risk_or_event"])
    parser.add_argument("--agent-compare-stride", type=int, default=50)
    parser.add_argument("--agent-risk-ttc-threshold", type=float, default=2.0)
    parser.add_argument("--radar-ablation", default="zero_bev", choices=["none", "zero_bev"])
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs" / "stage10_stress_campaign"))
    parser.add_argument("--report-root", default=str(PROJECT_ROOT / "benchmark" / "reports" / "stage10_stress_campaign"))
    parser.add_argument("--npc-handbrake", action="store_true", default=True)
    parser.add_argument("--blocker-distance-m", type=float, default=24.0)
    parser.add_argument("--adjacent-distance-m", type=float, default=42.0)
    return parser.parse_args()


def _run_command(command: List[str], *, label: str) -> None:
    print(f"[stress-campaign] {label}")
    print("  " + " ".join(command))
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))


def _case_specs(case_names: List[str]) -> List[Dict[str, str]]:
    supported = {
        "right": {"adjacent_side": "right", "case_label": "multilane_right"},
        "left": {"adjacent_side": "left", "case_label": "multilane_left"},
    }
    specs: List[Dict[str, str]] = []
    for case_name in case_names:
        key = case_name.strip().lower()
        if not key:
            continue
        if key not in supported:
            raise ValueError(f"Unsupported stress case '{case_name}'. Supported: {', '.join(sorted(supported))}")
        specs.append(supported[key])
    if not specs:
        raise ValueError("No stress cases selected.")
    return specs


def main() -> int:
    args = _parse_args()
    python_exe = sys.executable
    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    case_specs = _case_specs(str(args.cases).split(","))

    for repeat_idx in range(1, int(args.repeats) + 1):
        for spec in case_specs:
            case_label = spec["case_label"]
            run_id = f"{case_label}_run{repeat_idx}"
            manifest_path = output_root / f"{run_id}_manifest.json"
            stage10_output_root = output_root / run_id
            stage10_log_dir = report_root / run_id

            spawn_cmd = [
                python_exe,
                str(PROJECT_ROOT / "scripts" / "stage3_multilane_scenario.py"),
                "spawn",
                "--host", str(args.carla_host),
                "--port", str(args.carla_port),
                "--tm-port", str(args.tm_port),
                "--town", f"Carla/Maps/{args.map}",
                "--adjacent-side", spec["adjacent_side"],
                "--blocker-distance-m", str(args.blocker_distance_m),
                "--adjacent-distance-m", str(args.adjacent_distance_m),
                "--output-manifest", str(manifest_path),
            ]
            if bool(args.npc_handbrake):
                spawn_cmd.append("--npc-handbrake")

            stage10_cmd = [
                python_exe,
                str(PROJECT_ROOT / "scripts" / "run_stage10_stage1_live_bridge.py"),
                "--carla-root", str(args.carla_root),
                "--carla-host", str(args.carla_host),
                "--carla-port", str(args.carla_port),
                "--bev-repo", str(args.bev_repo),
                "--bev-config", str(args.bev_config),
                "--bev-ckpt", str(args.bev_ckpt),
                "--output-root", str(stage10_output_root),
                "--log-dir", str(stage10_log_dir),
                "--scenario-manifest", str(manifest_path),
                "--attach-autopilot",
                "--map", str(args.map),
                "--route-distance-m", str(args.route_distance_m),
                "--max-frames", str(args.max_frames),
                "--agent-mode", str(args.agent_mode),
                "--agent-trigger-mode", str(args.agent_trigger_mode),
                "--agent-compare-stride", str(args.agent_compare_stride),
                "--agent-risk-ttc-threshold", str(args.agent_risk_ttc_threshold),
                "--radar-ablation", str(args.radar_ablation),
            ]

            destroy_cmd = [
                python_exe,
                str(PROJECT_ROOT / "scripts" / "stage3_multilane_scenario.py"),
                "destroy",
                "--host", str(args.carla_host),
                "--port", str(args.carla_port),
                "--manifest", str(manifest_path),
            ]

            run_record: Dict[str, Any] = {
                "run_id": run_id,
                "case_label": case_label,
                "adjacent_side": spec["adjacent_side"],
                "repeat_index": repeat_idx,
                "manifest_path": str(manifest_path),
                "stage10_log_dir": str(stage10_log_dir),
                "stage10_output_root": str(stage10_output_root),
                "status": "pending",
                "started_at_utc": _utc_now_iso(),
            }
            runs.append(run_record)

            try:
                _run_command(spawn_cmd, label=f"spawn {run_id}")
                _run_command(stage10_cmd, label=f"stage10 {run_id}")
                run_record["status"] = "completed"
            except subprocess.CalledProcessError as exc:
                run_record["status"] = "failed"
                run_record["failed_command"] = exc.cmd
                run_record["exit_code"] = int(exc.returncode)
            finally:
                try:
                    if manifest_path.exists():
                        _run_command(destroy_cmd, label=f"destroy {run_id}")
                except subprocess.CalledProcessError as exc:
                    run_record["destroy_status"] = f"failed:{exc.returncode}"
                else:
                    run_record["destroy_status"] = "completed"
                run_record["finished_at_utc"] = _utc_now_iso()

            if run_record["status"] != "completed":
                break
        else:
            continue
        break

    summary = {
        "schema_version": "stage10_agent_stress_campaign_v1",
        "generated_at_utc": _utc_now_iso(),
        "map": args.map,
        "cases": [spec["case_label"] for spec in case_specs],
        "repeats": int(args.repeats),
        "max_frames": int(args.max_frames),
        "agent_mode": args.agent_mode,
        "runs": runs,
    }
    summary_path = report_root / f"campaign_summary_{int(time.time())}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[stress-campaign] summary -> {summary_path}")

    failed = [run for run in runs if run.get("status") != "completed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
