from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import (
    artifact_stage_origin,
    current_run_only,
    default_trace_requirements,
    expected_stage_chain,
    materialize_online_current_run_bundle,
    scenario_replay_artifacts,
)
from .io import dump_json, load_yaml, resolve_repo_path
from .provenance import TraceLogger, utc_now_iso, write_artifact_provenance


@dataclass
class CaseExecutionResult:
    scenario_id: str
    benchmark_mode: str
    trace_id: str
    status: str
    artifacts: dict[str, str]
    notes: list[str]
    source_updates: dict[str, Any] = field(default_factory=dict)
    stage_results: list[dict[str, Any]] = field(default_factory=list)
    artifact_provenance_path: str | None = None
    stage_chain_result_path: str | None = None


def _load_stage_profiles(stage_profiles_dir: Path) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for path in sorted(stage_profiles_dir.glob("*.yaml")):
        payload = load_yaml(path)
        profiles[str(payload.get("name") or path.stem)] = payload
    return profiles


def _to_cli_args(args_map: dict[str, Any], context: dict[str, Any]) -> list[str]:
    cli: list[str] = []
    for key, value in (args_map or {}).items():
        rendered = str(value).format(**context)
        rendered_lower = rendered.lower()
        if rendered_lower in {"true", "false"}:
            if rendered_lower == "true":
                cli.append(str(key))
            continue
        if rendered in {"", "None", "null"}:
            continue
        cli.extend([str(key), rendered])
    return cli


def _render_args(args_map: dict[str, Any], context: dict[str, Any]) -> dict[str, str]:
    rendered: dict[str, str] = {}
    for key, value in (args_map or {}).items():
        candidate = str(value).format(**context)
        if candidate in {"", "None", "null"}:
            continue
        rendered[str(key)] = candidate
    return rendered


def _run_stage_command(
    *,
    repo_root: Path,
    python_executable: str,
    runner_script: str,
    cli_args: list[str],
    log_path: Path,
) -> tuple[int, str]:
    command = [python_executable, str((repo_root / runner_script).resolve()), *cli_args]
    proc = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.write("EXIT_CODE: " + str(proc.returncode) + "\n\n")
        handle.write(proc.stdout or "")
        if proc.stderr:
            handle.write("\n[STDERR]\n")
            handle.write(proc.stderr)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output


def _execute_stage1_session_attach(
    *,
    repo_root: Path,
    case_run_dir: Path,
    profile: dict[str, Any],
    context: dict[str, Any],
) -> tuple[bool, list[str], dict[str, str]]:
    rendered = _render_args(profile.get("args") or {}, context)
    source_session_dir = resolve_repo_path(repo_root, rendered.get("source_session_dir") or context.get("stage1_session_dir"))
    required_files = list(profile.get("required_files") or ["session_summary.json", "live_summary.jsonl"])
    if source_session_dir is None:
        return False, ["stage1_session_attach_missing_source_session_dir"], {}

    missing = [name for name in required_files if not (source_session_dir / name).exists()]
    stage1_attach_dir = case_run_dir / "stage1"
    attach_manifest = {
        "runner_kind": str(profile.get("runner_kind", "stage1_session_attach")),
        "source_session_dir": str(source_session_dir),
        "required_files": required_files,
        "missing_files": missing,
        "materialized_at": utc_now_iso(),
    }
    dump_json(stage1_attach_dir / "stage1_session_attach.json", attach_manifest)
    if missing:
        return False, [f"stage1_session_attach_missing_files:{','.join(missing)}"], {
            "stage1_session_dir": str(source_session_dir),
        }
    return True, [f"stage1_session_attached:{source_session_dir}"], {
        "stage1_session_dir": str(source_session_dir),
    }


def _stage_artifact_paths(artifacts: dict[str, str], stage_name: str) -> list[str]:
    prefix = f"{stage_name}_"
    return [path for name, path in artifacts.items() if name.startswith(prefix)]


def execute_case(
    *,
    repo_root: Path,
    benchmark_run_id: str,
    case_spec: dict[str, Any],
    benchmark_mode: str,
    run_root: Path,
    python_executable: str | None = None,
    execute_stages: bool = True,
    stage_profiles_dir: Path,
    trace_logger: TraceLogger,
) -> CaseExecutionResult:
    scenario_id = str(case_spec["scenario_id"])
    trace_id = f"trace_{scenario_id}_{benchmark_run_id.split('_')[-1]}"
    case_run_dir = run_root / scenario_id
    case_run_dir.mkdir(parents=True, exist_ok=True)

    pyexe = python_executable or sys.executable
    profiles = _load_stage_profiles(stage_profiles_dir)
    source = case_spec.get("source") or {}
    route_profile = case_spec.get("route_profile") or {}
    scenario_recipe = case_spec.get("scenario_recipe") or {}
    stage_profile_map = case_spec.get("stage_profiles") or {}
    required_trace = default_trace_requirements(case_spec, benchmark_mode)
    chain = expected_stage_chain(case_spec, benchmark_mode)

    context = {
        "repo_root": str(repo_root),
        "case_id": scenario_id,
        "case_run_dir": str(case_run_dir),
        "scenario_manifest": scenario_recipe.get("scenario_manifest") or source.get("scenario_manifest") or "",
        "route_option": route_profile.get("route_option") or case_spec.get("route_option") or "keep_lane",
        "preferred_lane": route_profile.get("preferred_lane") or "current",
        "route_priority": route_profile.get("route_priority") or "normal",
        "stage1_session_dir": scenario_recipe.get("stage1_session_dir") or source.get("stage1_session_dir") or "",
        "stage2_output_dir": str(case_run_dir / "stage2"),
        "stage3a_output_dir": str(case_run_dir / "stage3a"),
        "stage3b_output_dir": str(case_run_dir / "stage3b"),
        "stage3c_output_dir": str(case_run_dir / "stage3c"),
        "stage4_output_dir": str(case_run_dir / "stage4"),
        "stage1_capture_output_root": str(case_run_dir / "stage1_capture"),
        "stage4_python_executable": case_spec.get("stage4_python_executable") or pyexe,
        "stage1_container_agent_root": os.environ.get("AGENTAI_STAGE1_CONTAINER_AGENT_ROOT", "/workspace/Agent-AI"),
        "stage1_device": os.environ.get("AGENTAI_STAGE1_DEVICE", "cuda"),
    }

    stage_results: list[dict[str, Any]] = []

    trace_logger.emit(
        case_id=scenario_id,
        trace_id=trace_id,
        benchmark_mode=benchmark_mode,
        step="spawn",
        status="started",
        stage_origin="spawn",
        stage_instance_id="spawn",
        current_run_only=(benchmark_mode != "replay_regression"),
        notes=[f"scenario_manifest={context['scenario_manifest']}"],
    )
    trace_logger.emit(
        case_id=scenario_id,
        trace_id=trace_id,
        benchmark_mode=benchmark_mode,
        step="spawn",
        status="success",
        stage_origin="spawn",
        stage_instance_id="spawn",
        current_run_only=(benchmark_mode != "replay_regression"),
        orchestration_status="spawned",
        notes=["spawn_or_attach_delegated_to_stage_runner"],
    )

    if not execute_stages:
        trace_logger.emit(
            case_id=scenario_id,
            trace_id=trace_id,
            benchmark_mode=benchmark_mode,
            step="evaluate",
            status="skipped",
            stage_origin="evaluate",
            stage_instance_id="evaluate",
            current_run_only=False,
            notes=["execute_stages=false"],
        )
        return CaseExecutionResult(
            scenario_id=scenario_id,
            benchmark_mode=benchmark_mode,
            trace_id=trace_id,
            status="skipped",
            artifacts={},
            notes=["stage_execution_skipped"],
        )

    if benchmark_mode == "replay_regression":
        trace_logger.emit(
            case_id=scenario_id,
            trace_id=trace_id,
            benchmark_mode=benchmark_mode,
            step="evaluate",
            status="success",
            stage_origin="evaluate",
            stage_instance_id="evaluate",
            current_run_only=False,
            notes=["replay_regression_no_stage_execution"],
        )
        return CaseExecutionResult(
            scenario_id=scenario_id,
            benchmark_mode=benchmark_mode,
            trace_id=trace_id,
            status="pass_through",
            artifacts={},
            notes=["replay_regression_no_stage_execution"],
        )

    stages_to_run = ["stage1", "stage2", "stage3a", "stage3b", "stage3c"] if benchmark_mode == "scenario_replay" else ["stage4"]

    for stage in stages_to_run:
        profile_name = stage_profile_map.get(stage)
        if not profile_name:
            return CaseExecutionResult(
                scenario_id=scenario_id,
                benchmark_mode=benchmark_mode,
                trace_id=trace_id,
                status="fail",
                artifacts={},
                notes=[f"missing_stage_profile:{stage}"],
                stage_results=stage_results,
            )
        profile = profiles.get(str(profile_name))
        if not profile:
            return CaseExecutionResult(
                scenario_id=scenario_id,
                benchmark_mode=benchmark_mode,
                trace_id=trace_id,
                status="fail",
                artifacts={},
                notes=[f"unknown_stage_profile:{profile_name}"],
                stage_results=stage_results,
            )

        stage_instance_id = f"{stage}:{profile_name}"
        stage_start = time.perf_counter()
        trace_logger.emit(
            case_id=scenario_id,
            trace_id=trace_id,
            benchmark_mode=benchmark_mode,
            step=stage,
            status="started",
            stage_origin=stage,
            stage_instance_id=stage_instance_id,
            current_run_only=(benchmark_mode == "online_e2e" and stage == "stage4"),
            notes=[f"profile={profile_name}"],
        )

        if str(profile.get("runner_kind")) == "stage1_session_attach":
            success, notes, updates = _execute_stage1_session_attach(
                repo_root=repo_root,
                case_run_dir=case_run_dir,
                profile=profile,
                context=context,
            )
            context.update(updates)
            timing_ms = (time.perf_counter() - stage_start) * 1000.0
            stage_result = {
                "stage": stage,
                "profile": profile_name,
                "status": "success" if success else "fail",
                "runner_kind": "stage1_session_attach",
                "timing_ms": float(timing_ms),
                "notes": notes,
            }
            stage_results.append(stage_result)
            if not success:
                trace_logger.emit(
                    case_id=scenario_id,
                    trace_id=trace_id,
                    benchmark_mode=benchmark_mode,
                    step=stage,
                    status="fail",
                    stage_origin=stage,
                    stage_instance_id=stage_instance_id,
                    current_run_only=False,
                    timing_ms=float(timing_ms),
                    failure_reason=";".join(notes),
                    notes=notes,
                )
                return CaseExecutionResult(
                    scenario_id=scenario_id,
                    benchmark_mode=benchmark_mode,
                    trace_id=trace_id,
                    status="fail",
                    artifacts={},
                    notes=[f"stage_failed:{stage}"],
                    source_updates={"stage1_session_dir": context["stage1_session_dir"]},
                    stage_results=stage_results,
                )
            trace_logger.emit(
                case_id=scenario_id,
                trace_id=trace_id,
                benchmark_mode=benchmark_mode,
                step=stage,
                status="success",
                stage_origin=stage,
                stage_instance_id=stage_instance_id,
                current_run_only=False,
                timing_ms=float(timing_ms),
                orchestration_status="attached_existing_session",
                notes=notes,
            )
            continue

        cli_args = _to_cli_args(profile.get("args") or {}, context)
        stage_python_executable = str(profile.get("python_executable") or pyexe).format(**context)
        exit_code, output = _run_stage_command(
            repo_root=repo_root,
            python_executable=stage_python_executable,
            runner_script=str(profile["runner_script"]),
            cli_args=cli_args,
            log_path=case_run_dir / "logs" / f"{stage}.log",
        )
        timing_ms = (time.perf_counter() - stage_start) * 1000.0
        stage_result = {
            "stage": stage,
            "profile": profile_name,
            "status": "success" if exit_code == 0 else "fail",
            "runner_script": str(profile["runner_script"]),
            "timing_ms": float(timing_ms),
            "log_path": str(case_run_dir / "logs" / f"{stage}.log"),
        }
        stage_results.append(stage_result)
        if exit_code != 0:
            trace_logger.emit(
                case_id=scenario_id,
                trace_id=trace_id,
                benchmark_mode=benchmark_mode,
                step=stage,
                status="fail",
                stage_origin=stage,
                stage_instance_id=stage_instance_id,
                current_run_only=(benchmark_mode == "online_e2e" and stage == "stage4"),
                timing_ms=float(timing_ms),
                failure_reason=f"exit_code={exit_code}",
                notes=[f"exit_code={exit_code}", output[-500:]],
            )
            return CaseExecutionResult(
                scenario_id=scenario_id,
                benchmark_mode=benchmark_mode,
                trace_id=trace_id,
                status="fail",
                artifacts={},
                notes=[f"stage_failed:{stage}"],
                stage_results=stage_results,
            )
        trace_logger.emit(
            case_id=scenario_id,
            trace_id=trace_id,
            benchmark_mode=benchmark_mode,
            step=stage,
            status="success",
            stage_origin=stage,
            stage_instance_id=stage_instance_id,
            current_run_only=(benchmark_mode == "online_e2e" and stage == "stage4"),
            timing_ms=float(timing_ms),
            notes=[f"profile={profile_name}"],
        )

    source_updates: dict[str, Any]
    materialization_notes: list[str] = []
    if benchmark_mode == "scenario_replay":
        artifacts, source_updates = scenario_replay_artifacts(
            case_run_dir=case_run_dir,
            stage1_session_dir=Path(context["stage1_session_dir"]) if context.get("stage1_session_dir") else None,
        )
    else:
        artifacts, source_updates, materialization_notes = materialize_online_current_run_bundle(
            case_run_dir=case_run_dir,
            case_spec=case_spec,
        )
        for stage_name in ["stage1", "stage2", "stage3a", "stage3b", "stage3c"]:
            stage_notes = [f"materialized_from_stage4_output:{stage_name}"]
            if materialization_notes:
                stage_notes.extend(materialization_notes)
            trace_logger.emit(
                case_id=scenario_id,
                trace_id=trace_id,
                benchmark_mode=benchmark_mode,
                step=stage_name,
                status="success",
                stage_origin=stage_name,
                stage_instance_id=f"{stage_name}:materialized",
                current_run_only=True,
                orchestration_status="materialized_from_stage4_output",
                artifact_paths=_stage_artifact_paths(artifacts, stage_name),
                notes=stage_notes,
            )
            stage_results.append(
                {
                    "stage": stage_name,
                    "profile": "materialized_from_stage4_output",
                    "status": "success",
                    "timing_ms": None,
                    "notes": stage_notes,
                }
            )

    provenance_rows = []
    for artifact_name, path in artifacts.items():
        provenance_rows.append(
            {
                "artifact_name": artifact_name,
                "path": path,
                "stage_origin": artifact_stage_origin(artifact_name, benchmark_mode),
                "stage_instance_id": artifact_stage_origin(artifact_name, benchmark_mode),
                "created_at": utc_now_iso(),
                "source_runner": benchmark_mode,
                "current_run_only": current_run_only(case_run_dir, path),
                "exists": Path(path).exists(),
                "notes": materialization_notes if benchmark_mode == "online_e2e" else [],
            }
        )
    artifact_provenance_path = case_run_dir / "artifact_provenance.json"
    write_artifact_provenance(
        output_path=artifact_provenance_path,
        benchmark_run_id=benchmark_run_id,
        case_id=scenario_id,
        trace_id=trace_id,
        artifacts=provenance_rows,
    )

    stage_chain_result_path = case_run_dir / "stage_chain_result.json"
    dump_json(
        stage_chain_result_path,
        {
            "benchmark_run_id": benchmark_run_id,
            "case_id": scenario_id,
            "trace_id": trace_id,
            "benchmark_mode": benchmark_mode,
            "expected_stage_chain": chain,
            "trace_requirements": required_trace,
            "status": "success",
            "stages": stage_results,
            "artifacts_ready": sorted(artifacts.keys()),
        },
    )

    trace_logger.emit(
        case_id=scenario_id,
        trace_id=trace_id,
        benchmark_mode=benchmark_mode,
        step="evaluate",
        status="success",
        stage_origin="evaluate",
        stage_instance_id="evaluate",
        current_run_only=(benchmark_mode != "replay_regression"),
        orchestration_status="artifacts_materialized",
        artifact_paths=list(artifacts.values()),
        notes=["artifacts_ready_for_benchmark_evaluation", *materialization_notes],
    )
    return CaseExecutionResult(
        scenario_id=scenario_id,
        benchmark_mode=benchmark_mode,
        trace_id=trace_id,
        status="success",
        artifacts=artifacts,
        notes=materialization_notes,
        source_updates=source_updates,
        stage_results=stage_results,
        artifact_provenance_path=str(artifact_provenance_path),
        stage_chain_result_path=str(stage_chain_result_path),
    )
