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
    parser.add_argument("--cases", default="full",
                        help="Comma-separated deterministic stress cases to run. Supported aliases: full, lane_only")
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
    parser.add_argument("--spawn-probe-top-k", type=int, default=8,
                        help="Number of corridor candidates to probe for each deterministic spawn case")
    parser.add_argument("--spawn-max-candidates", type=int, default=6,
                        help="Maximum number of probed candidates to try before giving up on a case")
    return parser.parse_args()


def _run_command(command: List[str], *, label: str, check: bool = True) -> subprocess.CompletedProcess[Any]:
    t0 = time.monotonic()
    print(f"[stress-campaign] {label}")
    print("  " + " ".join(command))
    result = subprocess.run(command, check=check, cwd=str(PROJECT_ROOT))
    print(f"[stress-campaign] {label} finished in {time.monotonic() - t0:.1f}s")
    return result


def _build_summary_payload(
    *,
    args: argparse.Namespace,
    case_specs: List[Dict[str, Any]],
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": "stage10_agent_stress_campaign_v1",
        "generated_at_utc": _utc_now_iso(),
        "map": args.map,
        "cases": [spec["case_label"] for spec in case_specs],
        "repeats": int(args.repeats),
        "max_frames": int(args.max_frames),
        "agent_mode": args.agent_mode,
        "runs": runs,
    }


def _persist_progress(
    *,
    report_root: Path,
    args: argparse.Namespace,
    case_specs: List[Dict[str, Any]],
    runs: List[Dict[str, Any]],
) -> Path:
    payload = _build_summary_payload(args=args, case_specs=case_specs, runs=runs)
    latest_path = report_root / "campaign_latest.json"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return latest_path


def _spawn_with_probe_retry(
    *,
    python_exe: str,
    args: argparse.Namespace,
    spec: Dict[str, Any],
    run_id: str,
    manifest_path: Path,
) -> Dict[str, Any]:
    probe_path = manifest_path.with_suffix(".probe.json")
    if manifest_path.exists():
        manifest_path.unlink()
    if probe_path.exists():
        probe_path.unlink()

    probe_cmd = [
        python_exe,
        str(PROJECT_ROOT / "scripts" / "stage3_multilane_scenario.py"),
        "--host", str(args.carla_host),
        "--port", str(args.carla_port),
        "--tm-port", str(args.tm_port),
        "--town", f"Carla/Maps/{args.map}",
        "probe",
        "--top-k", str(int(args.spawn_probe_top_k)),
        "--adjacent-side", str(spec["adjacent_side"]),
        "--output-json", str(probe_path),
    ]
    _run_command(probe_cmd, label=f"probe {run_id}")

    candidates = json.loads(probe_path.read_text(encoding="utf-8"))
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"No spawn candidates found for {run_id}")

    max_candidates = max(1, min(int(args.spawn_max_candidates), len(candidates)))
    last_error: str | None = None
    for rank in range(max_candidates):
        candidate = dict(candidates[rank])
        if manifest_path.exists():
            manifest_path.unlink()
        spawn_cmd = [
            python_exe,
            str(PROJECT_ROOT / "scripts" / "stage3_multilane_scenario.py"),
            "--host", str(args.carla_host),
            "--port", str(args.carla_port),
            "--tm-port", str(args.tm_port),
            "--town", f"Carla/Maps/{args.map}",
            "spawn",
            "--adjacent-side", str(spec["adjacent_side"]),
            "--road-id", str(int(candidate["road_id"])),
            "--section-id", str(int(candidate["section_id"])),
            "--lane-id", str(int(candidate["lane_id"])),
            "--s", str(float(candidate["s"])),
            "--blocker-distance-m", str(float(spec.get("blocker_distance_m", args.blocker_distance_m))),
            "--adjacent-distance-m", str(float(spec.get("adjacent_distance_m", args.adjacent_distance_m))),
            "--output-manifest", str(manifest_path),
        ]
        if bool(args.npc_handbrake):
            spawn_cmd.append("--npc-handbrake")

        print(
            f"[stress-campaign] trying spawn candidate {rank + 1}/{max_candidates} "
            f"for {run_id} road={candidate['road_id']} lane={candidate['lane_id']} s={float(candidate['s']):.2f}"
        )
        result = _run_command(spawn_cmd, label=f"spawn {run_id} candidate={rank + 1}", check=False)
        if result.returncode == 0 and manifest_path.exists():
            return {
                "candidate_rank": rank,
                "candidate": candidate,
                "probe_path": str(probe_path),
            }

        last_error = f"spawn returncode={result.returncode}"
        print(f"[stress-campaign] candidate {rank + 1} failed for {run_id} -> {last_error}")
        time.sleep(1.0)

    raise RuntimeError(f"Failed to spawn {run_id} after {max_candidates} candidates. last_error={last_error}")


def _case_specs(case_names: List[str]) -> List[Dict[str, Any]]:
    supported: Dict[str, Dict[str, Any]] = {
        "right": {
            "adjacent_side": "right",
            "case_label": "multilane_right_nominal",
            "blocker_distance_m": 24.0,
            "adjacent_distance_m": 42.0,
            "route_distance_m": 120.0,
        },
        "left": {
            "adjacent_side": "left",
            "case_label": "multilane_left_nominal",
            "blocker_distance_m": 24.0,
            "adjacent_distance_m": 42.0,
            "route_distance_m": 120.0,
        },
        "unsafe_right": {
            "adjacent_side": "right",
            "case_label": "unsafe_lane_change_right",
            "blocker_distance_m": 18.0,
            "adjacent_distance_m": 12.0,
            "route_distance_m": 90.0,
        },
        "unsafe_left": {
            "adjacent_side": "left",
            "case_label": "unsafe_lane_change_left",
            "blocker_distance_m": 18.0,
            "adjacent_distance_m": 12.0,
            "route_distance_m": 90.0,
        },
        "stop_follow": {
            "adjacent_side": "right",
            "case_label": "stop_follow_ambiguity",
            "blocker_distance_m": 10.0,
            "adjacent_distance_m": 14.0,
            "route_distance_m": 60.0,
        },
        "emergency_brake": {
            "adjacent_side": "right",
            "case_label": "emergency_brake_conflict",
            "blocker_distance_m": 6.0,
            "adjacent_distance_m": 16.0,
            "route_distance_m": 45.0,
        },
    }
    aliases = {
        "full": ["right", "left", "unsafe_right", "unsafe_left", "stop_follow", "emergency_brake"],
        "lane_only": ["right", "left"],
    }

    expanded: List[str] = []
    for case_name in case_names:
        key = case_name.strip().lower()
        if not key:
            continue
        expanded.extend(aliases.get(key, [key]))

    specs: List[Dict[str, Any]] = []
    for key in expanded:
        if key not in supported:
            labels = ", ".join(sorted({*supported.keys(), *aliases.keys()}))
            raise ValueError(f"Unsupported stress case '{key}'. Supported: {labels}")
        specs.append(dict(supported[key]))
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
    total_runs = len(case_specs) * max(1, int(args.repeats))

    run_counter = 0
    for repeat_idx in range(1, int(args.repeats) + 1):
        for spec in case_specs:
            run_counter += 1
            case_label = spec["case_label"]
            run_id = f"{case_label}_run{repeat_idx}"
            manifest_path = output_root / f"{run_id}_manifest.json"
            stage10_output_root = output_root / run_id
            stage10_log_dir = report_root / run_id

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
                "--route-distance-m", str(float(spec.get("route_distance_m", args.route_distance_m))),
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
                "--host", str(args.carla_host),
                "--port", str(args.carla_port),
                "destroy",
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
                "blocker_distance_m": float(spec.get("blocker_distance_m", args.blocker_distance_m)),
                "adjacent_distance_m": float(spec.get("adjacent_distance_m", args.adjacent_distance_m)),
                "route_distance_m": float(spec.get("route_distance_m", args.route_distance_m)),
                "status": "pending",
                "started_at_utc": _utc_now_iso(),
            }
            runs.append(run_record)
            latest_path = _persist_progress(
                report_root=report_root,
                args=args,
                case_specs=case_specs,
                runs=runs,
            )

            print(
                f"[stress-campaign] progress {run_counter}/{total_runs} "
                f"repeat={repeat_idx}/{int(args.repeats)} case={case_label}"
            )
            print(f"[stress-campaign] latest summary -> {latest_path}")

            try:
                spawn_meta = _spawn_with_probe_retry(
                    python_exe=python_exe,
                    args=args,
                    spec=spec,
                    run_id=run_id,
                    manifest_path=manifest_path,
                )
                run_record["spawn_probe"] = spawn_meta
                _run_command(stage10_cmd, label=f"stage10 {run_id}")
                run_record["status"] = "completed"
            except Exception as exc:
                run_record["status"] = "failed"
                if isinstance(exc, subprocess.CalledProcessError):
                    run_record["failed_command"] = exc.cmd
                    run_record["exit_code"] = int(exc.returncode)
                run_record["error"] = str(exc)
            finally:
                try:
                    if manifest_path.exists():
                        _run_command(destroy_cmd, label=f"destroy {run_id}")
                except subprocess.CalledProcessError as exc:
                    run_record["destroy_status"] = f"failed:{exc.returncode}"
                else:
                    run_record["destroy_status"] = "completed"
                run_record["finished_at_utc"] = _utc_now_iso()
                _persist_progress(
                    report_root=report_root,
                    args=args,
                    case_specs=case_specs,
                    runs=runs,
                )

            if run_record["status"] != "completed":
                break
        else:
            continue
        break

    summary = _build_summary_payload(args=args, case_specs=case_specs, runs=runs)
    summary_path = report_root / f"campaign_summary_{int(time.time())}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[stress-campaign] summary -> {summary_path}")

    failed = [run for run in runs if run.get("status") != "completed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
