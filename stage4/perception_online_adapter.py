from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List

from stage2.prediction_loader import (
    Stage1FrameArtifact,
    _resolve_artifact_path,
    load_normalized_prediction,
)

from .state_store import PerceptionFrameResult

LOGGER = logging.getLogger("stage4.perception_online_adapter")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _seconds_to_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) * 1000.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_tail(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return "".join(lines[-max_lines:])


class PerceptionOnlineAdapter:
    def __init__(
        self,
        *,
        project_root: str | Path,
        samples_root: str | Path,
        stage1_output_root: str | Path,
        session_name: str,
        num_samples: int,
        min_score: float,
        launch_mode: str = "docker_watch",
        external_session_root: str | Path | None = None,
        container_name: str = "bevfusion-cu128",
        container_agent_root: str = "/workspace/Agent-AI",
        container_repo_root: str = "/workspace/Bevfusion-Z",
        container_config: str = "/workspace/Bevfusion-Z/configs/nuscenes/det/transfusion/secfpn/camera+lidar+radar/swint_v0p075/sefuser_radarbev.yaml",
        container_checkpoint: str = "/workspace/Data-Train/epoch_8_3sensor.pth",
        device: str = "cuda",
        radar_bridge: str = "minimal",
        radar_ablation: str = "zero_bev",
        compare_zero_radar: bool = False,
        adapter_lidar_sweeps_test: int = 3,
        adapter_radar_sweeps: int = 3,
        adapter_lidar_max_points: int = 120000,
        adapter_radar_max_points: int = 1200,
        prediction_variant: str = "auto",
        poll_interval_seconds: float = 0.25,
        idle_timeout_seconds: float = 0.0,
        external_watch_start_sequence_index: int = -1,
        external_watch_start_sample_name: str | None = None,
        external_watch_end_sequence_index: int = -1,
        external_watch_end_sample_name: str | None = None,
        external_watch_max_payloads_per_poll: int = 0,
        worker_log_path: str | Path | None = None,
        worker_audit_path: str | Path | None = None,
        worker_failure_trace_path: str | Path | None = None,
        worker_health_path: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.samples_root = Path(samples_root).resolve()
        self.stage1_output_root = Path(stage1_output_root).resolve()
        self.session_name = str(session_name)
        self.num_samples = int(num_samples)
        self.min_score = float(min_score)
        self.launch_mode = str(launch_mode)
        self.external_session_root = (
            None if external_session_root is None else Path(external_session_root).resolve()
        )
        self.container_name = str(container_name)
        self.container_agent_root = str(container_agent_root)
        self.container_repo_root = str(container_repo_root)
        self.container_config = str(container_config)
        self.container_checkpoint = str(container_checkpoint)
        self.device = str(device)
        self.radar_bridge = str(radar_bridge)
        self.radar_ablation = str(radar_ablation)
        self.compare_zero_radar = bool(compare_zero_radar)
        self.adapter_lidar_sweeps_test = int(adapter_lidar_sweeps_test)
        self.adapter_radar_sweeps = int(adapter_radar_sweeps)
        self.adapter_lidar_max_points = int(adapter_lidar_max_points)
        self.adapter_radar_max_points = int(adapter_radar_max_points)
        self.prediction_variant = str(prediction_variant)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.idle_timeout_seconds = float(idle_timeout_seconds)
        self.external_watch_start_sequence_index = int(external_watch_start_sequence_index)
        self.external_watch_start_sample_name = str(external_watch_start_sample_name or "")
        self.external_watch_end_sequence_index = int(external_watch_end_sequence_index)
        self.external_watch_end_sample_name = str(external_watch_end_sample_name or "")
        self.external_watch_max_payloads_per_poll = int(external_watch_max_payloads_per_poll)
        self.worker_log_path = (
            None if worker_log_path is None else Path(worker_log_path).resolve()
        )
        self.worker_audit_path = (
            None if worker_audit_path is None else Path(worker_audit_path).resolve()
        )
        self.worker_failure_trace_path = (
            None if worker_failure_trace_path is None else Path(worker_failure_trace_path).resolve()
        )
        self.worker_health_path = (
            None if worker_health_path is None else Path(worker_health_path).resolve()
        )

        self.session_root = (
            self.external_session_root
            if self.launch_mode == "external_watch"
            else self.stage1_output_root / self.session_name
        )
        self.summary_path = self.session_root / "live_summary.jsonl"
        self.session_manifest_path = self.session_root / "session_manifest.json"

        self._summary_offset = 0
        self._stage1_process: subprocess.Popen[str] | None = None
        self._worker_exit_code: int | None = None
        self._worker_log_handle: Any | None = None
        self._capture_registry: Dict[str, Dict[str, Any]] = {}
        self._processed_samples: set[str] = set()
        self._history_skipped_samples = 0
        self._inferred_samples = 0
        self._spawn_command: List[str] = []
        self._last_seen_sample: str | None = None
        self._last_emitted_prediction: str | None = None
        self._last_summary_payload: Dict[str, Any] | None = None
        self._last_heartbeat_utc: str | None = None
        self._worker_failure_class: str = "unknown"
        self._worker_root_cause: str | None = None
        self._pending_summary_payloads: List[Dict[str, Any]] = []

    def _effective_external_watch_start_index(self) -> int | None:
        if self.external_watch_start_sequence_index >= 0:
            return int(self.external_watch_start_sequence_index)
        if self.external_watch_start_sample_name:
            match = re.search(r"sample_(\d+)", self.external_watch_start_sample_name)
            if match:
                return int(match.group(1))
        return None

    def _effective_external_watch_end_index(self) -> int | None:
        if self.external_watch_end_sequence_index >= 0:
            return int(self.external_watch_end_sequence_index)
        if self.external_watch_end_sample_name:
            match = re.search(r"sample_(\d+)", self.external_watch_end_sample_name)
            if match:
                return int(match.group(1))
        return None

    def _emit_health(self, state: str, *, current_sample: str | None = None, notes: List[str] | None = None) -> None:
        path = self.worker_health_path
        if path is None:
            return
        _append_jsonl(
            path,
            {
                "timestamp": _utc_now_iso(),
                "worker_state": str(state),
                "current_sample": current_sample,
                "last_successful_sample": self._last_seen_sample,
                "last_successful_prediction": self._last_emitted_prediction,
                "notes": list(notes or []),
            },
        )
        self._last_heartbeat_utc = _utc_now_iso()

    def _docker_container_state(self) -> Dict[str, Any]:
        command = ["docker", "inspect", "-f", "{{.State.Status}}", self.container_name]
        proc = subprocess.run(
            command,
            cwd=str(self.project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        status = (proc.stdout or "").strip()
        return {
            "command": command,
            "exit_code": int(proc.returncode),
            "status": status,
            "stderr": (proc.stderr or "").strip(),
        }

    def _ensure_container_running(self) -> None:
        if self.launch_mode != "docker_watch":
            return
        state = self._docker_container_state()
        if state["exit_code"] != 0:
            self._worker_failure_class = "path_mapping_error"
            self._worker_root_cause = f"docker_inspect_failed:{state['stderr'] or 'unknown'}"
            self._emit_health("error", notes=[self._worker_root_cause])
            raise RuntimeError(
                f"Unable to inspect container '{self.container_name}': {state['stderr'] or 'unknown error'}"
            )
        if state["status"] == "running":
            return
        start_command = ["docker", "start", self.container_name]
        proc = subprocess.run(
            start_command,
            cwd=str(self.project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self._worker_failure_class = "crash_on_boot"
            self._worker_root_cause = f"container_not_running_and_start_failed:{(proc.stderr or '').strip()}"
            self._emit_health("error", notes=[self._worker_root_cause])
            raise RuntimeError(
                f"Container '{self.container_name}' is not running and could not be started: {(proc.stderr or '').strip()}"
            )
        self._emit_health("booting", notes=[f"auto_started_container:{self.container_name}"])

    def _classify_failure(self, *, worker_exit_code: int | None) -> tuple[str, str]:
        if worker_exit_code is None:
            return "unknown", "worker_exit_code_unavailable"
        if worker_exit_code == 0:
            return "none", "clean_exit"

        tail = ""
        if self.worker_log_path is not None:
            tail = _read_tail(self.worker_log_path, max_lines=200)
        tail_lower = tail.lower()

        if "container" in tail_lower and "is not running" in tail_lower:
            return "crash_on_boot", "docker_container_not_running"
        if "no such file or directory" in tail_lower and ("samples-root" in tail_lower or "output-root" in tail_lower):
            return "path_mapping_error", "host_container_path_mapping_error"
        if "filenotfounderror" in tail_lower and "sample" in tail_lower:
            return "bad_sample", "missing_sample_artifact"
        if "traceback" in tail_lower and self._inferred_samples == 0:
            return "crash_on_boot", "python_exception_before_first_infer"
        if "traceback" in tail_lower and self._inferred_samples > 0:
            return "crash_on_infer", "python_exception_during_infer"
        if self._inferred_samples == 0 and len(self._capture_registry) > 0:
            return "no_input_detected", "no_prediction_emitted"
        return "unknown", "nonzero_exit_without_signature"

    def _write_worker_audit(self) -> None:
        worker_exit_code = self._worker_exit_code
        if self._stage1_process is not None:
            worker_exit_code = self._stage1_process.poll()

        failure_class, root_cause = self._classify_failure(worker_exit_code=worker_exit_code)
        self._worker_failure_class = failure_class
        self._worker_root_cause = root_cause

        tail = ""
        if self.worker_log_path is not None:
            tail = _read_tail(self.worker_log_path, max_lines=240)

        if self.worker_failure_trace_path is not None:
            self.worker_failure_trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.worker_failure_trace_path.write_text(tail, encoding="utf-8")

        if self.worker_audit_path is None:
            return
        self.worker_audit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "spawn_command": " ".join(self._spawn_command),
            "python_executable": None if not self._spawn_command else self._spawn_command[-1],
            "cwd": str(self.project_root),
            "worker_exit_code": worker_exit_code,
            "failure_class": failure_class,
            "last_seen_sample": self._last_seen_sample,
            "last_emitted_prediction": self._last_emitted_prediction,
            "last_successful_heartbeat": self._last_heartbeat_utc,
            "root_cause": root_cause,
            "notes": [
                f"launch_mode={self.launch_mode}",
                f"observed_captures={len(self._processed_samples) + len(self._capture_registry)}",
                f"inferred_samples={self._inferred_samples}",
                f"history_skipped_samples={self._history_skipped_samples}",
                f"unresolved_captures={len(self._capture_registry)}",
            ],
        }
        with self.worker_audit_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _host_path_to_container(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(
                f"Path {path} is outside project root {self.project_root}; "
                "cannot map it into the BEVFusion container workspace."
            ) from exc
        return str(PurePosixPath(self.container_agent_root) / PurePosixPath(*relative.parts))

    def _build_watch_command(self) -> List[str]:
        container_samples_root = self._host_path_to_container(self.samples_root)
        container_output_root = self._host_path_to_container(self.stage1_output_root)
        python_parts = [
            "python -u",
            shlex.quote(f"{self.container_agent_root}/scripts/live_infer.py"),
            "--samples-root",
            shlex.quote(container_samples_root),
            "--repo-root",
            shlex.quote(self.container_repo_root),
            "--config",
            shlex.quote(self.container_config),
            "--checkpoint",
            shlex.quote(self.container_checkpoint),
            "--output-root",
            shlex.quote(container_output_root),
            "--session-name",
            shlex.quote(self.session_name),
            "--num-samples",
            shlex.quote(str(self.num_samples)),
            "--device",
            shlex.quote(self.device),
            "--adapter-lidar-sweeps-test",
            shlex.quote(str(self.adapter_lidar_sweeps_test)),
            "--adapter-radar-sweeps",
            shlex.quote(str(self.adapter_radar_sweeps)),
            "--adapter-lidar-max-points",
            shlex.quote(str(self.adapter_lidar_max_points)),
            "--adapter-radar-max-points",
            shlex.quote(str(self.adapter_radar_max_points)),
            "--radar-bridge",
            shlex.quote(self.radar_bridge),
            "--radar-ablation",
            shlex.quote(self.radar_ablation),
            "--poll-interval-seconds",
            shlex.quote(str(self.poll_interval_seconds)),
            "--idle-timeout-seconds",
            shlex.quote(str(self.idle_timeout_seconds)),
        ]
        if self.compare_zero_radar:
            python_parts.append("--compare-zero-radar")
        else:
            python_parts.append("--no-compare-zero-radar")
        bash_command = " ".join(
            [
                "source /root/miniconda3/bin/activate bevfusion",
                "&&",
                *python_parts,
            ]
        )
        return ["docker", "exec", self.container_name, "/bin/bash", "-lc", bash_command]

    def start(self) -> None:
        self.stage1_output_root.mkdir(parents=True, exist_ok=True)
        self._emit_health("booting", notes=[f"launch_mode={self.launch_mode}"])
        if self.launch_mode == "external_watch":
            LOGGER.info("Using external Stage1 watch session root=%s", self.session_root)
            self._emit_health("watching", notes=["external_watch_mode"])
            return
        if self.launch_mode != "docker_watch":
            raise ValueError(
                f"Unsupported Stage1 launch mode {self.launch_mode!r}. "
                "Supported modes: docker_watch, external_watch."
            )
        self._ensure_container_running()
        if self.session_root.exists():
            raise FileExistsError(f"Stage1 watch session already exists: {self.session_root}")

        command = self._build_watch_command()
        self._spawn_command = list(command)
        LOGGER.info("Launching Stage1 watch worker via docker: %s", " ".join(command))

        stdout_handle = None
        if self.worker_log_path is not None:
            self.worker_log_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = self.worker_log_path.open("w", encoding="utf-8")
            self._worker_log_handle = stdout_handle
        try:
            self._stage1_process = subprocess.Popen(
                command,
                cwd=str(self.project_root),
                stdout=stdout_handle or subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self._emit_health("watching", notes=["worker_process_started"])
        except Exception:
            if stdout_handle is not None:
                stdout_handle.close()
            self._emit_health("error", notes=["worker_process_start_failed"])
            raise

    def register_captured_sample(
        self,
        *,
        sample_name: str,
        tick_id: int,
        carla_frame: int,
        capture_monotonic_s: float,
    ) -> None:
        self._capture_registry[str(sample_name)] = {
            "tick_id": int(tick_id),
            "carla_frame": int(carla_frame),
            "capture_monotonic_s": float(capture_monotonic_s),
        }

    def _read_new_summary_payloads(self) -> List[Dict[str, Any]]:
        if not self.summary_path.exists():
            return []
        payloads: List[Dict[str, Any]] = []
        with self.summary_path.open("r", encoding="utf-8") as handle:
            handle.seek(self._summary_offset)
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                payloads.append(json.loads(line))
            self._summary_offset = handle.tell()
        return payloads

    def _artifact_from_payload(self, payload: Dict[str, Any]) -> Stage1FrameArtifact:
        status = str(payload["status"])
        if status == "inferred_compare":
            variant = "bridge_minimal" if self.prediction_variant == "auto" else self.prediction_variant
            prediction_dir = _resolve_artifact_path(
                payload["comparison_output_dir"],
                session_dir=self.session_root,
            ) / variant
        else:
            variant = "single"
            prediction_dir = _resolve_artifact_path(
                payload["output_dir"],
                session_dir=self.session_root,
            )
        return Stage1FrameArtifact(
            sample_name=str(payload["sample_name"]),
            sequence_index=int(payload["sequence_index"]),
            frame_id=int(payload["frame_id"]),
            timestamp=float(payload["timestamp"]),
            source_sample_dir=_resolve_artifact_path(
                payload["source_sample_dir"],
                session_dir=self.session_root,
            ),
            source_prediction_dir=prediction_dir,
            prediction_variant=variant,
            stage1_status=status,
        )

    def poll_predictions(self) -> List[PerceptionFrameResult]:
        results: List[PerceptionFrameResult] = []
        external_watch_start_index = self._effective_external_watch_start_index()
        external_watch_end_index = self._effective_external_watch_end_index()
        new_payloads = self._read_new_summary_payloads()
        if new_payloads:
            self._pending_summary_payloads.extend(new_payloads)
        max_payloads_per_poll = (
            self.external_watch_max_payloads_per_poll
            if self.launch_mode == "external_watch" and self.external_watch_max_payloads_per_poll > 0
            else 0
        )
        pending_payloads = list(self._pending_summary_payloads)
        self._pending_summary_payloads = []
        for payload in pending_payloads:
            if max_payloads_per_poll and len(results) >= int(max_payloads_per_poll):
                self._pending_summary_payloads.append(payload)
                continue
            sample_name = str(payload.get("sample_name"))
            self._last_seen_sample = sample_name
            self._last_summary_payload = dict(payload)
            if sample_name in self._processed_samples:
                continue
            if (
                self.launch_mode == "external_watch"
                and external_watch_start_index is not None
                and int(payload.get("sequence_index", -1)) < int(external_watch_start_index)
            ):
                self._processed_samples.add(sample_name)
                self._capture_registry.pop(sample_name, None)
                self._history_skipped_samples += 1
                self._emit_health(
                    "idle",
                    current_sample=sample_name,
                    notes=[f"before_external_watch_start_index={external_watch_start_index}"],
                )
                continue
            if (
                self.launch_mode == "external_watch"
                and external_watch_end_index is not None
                and int(payload.get("sequence_index", -1)) > int(external_watch_end_index)
            ):
                self._processed_samples.add(sample_name)
                self._capture_registry.pop(sample_name, None)
                self._emit_health(
                    "idle",
                    current_sample=sample_name,
                    notes=[f"after_external_watch_end_index={external_watch_end_index}"],
                )
                continue

            status = str(payload.get("status"))
            if status == "skipped_history_not_ready":
                self._processed_samples.add(sample_name)
                self._capture_registry.pop(sample_name, None)
                self._history_skipped_samples += 1
                self._emit_health("idle", current_sample=sample_name, notes=["history_not_ready"])
                continue
            if status not in {"inferred", "inferred_compare"}:
                continue

            artifact = self._artifact_from_payload(payload)
            normalized = load_normalized_prediction(artifact, min_score=self.min_score)
            capture_meta = self._capture_registry.pop(sample_name, None)
            perception_latency_ms = None
            if capture_meta is not None:
                perception_latency_ms = (time.monotonic() - float(capture_meta["capture_monotonic_s"])) * 1000.0

            result = PerceptionFrameResult(
                sample_name=artifact.sample_name,
                sequence_index=artifact.sequence_index,
                frame_id=artifact.frame_id,
                timestamp=artifact.timestamp,
                capture_tick_id=None if capture_meta is None else int(capture_meta["tick_id"]),
                capture_carla_frame=None if capture_meta is None else int(capture_meta["carla_frame"]),
                capture_monotonic_s=None if capture_meta is None else float(capture_meta["capture_monotonic_s"]),
                source_sample_dir=str(artifact.source_sample_dir),
                source_prediction_dir=str(artifact.source_prediction_dir),
                stage1_status=artifact.stage1_status,
                prediction_variant=artifact.prediction_variant,
                adapt_latency_ms=_seconds_to_ms(payload.get("adapt_seconds")),
                infer_latency_ms=_seconds_to_ms(payload.get("infer_seconds")),
                visualize_latency_ms=_seconds_to_ms(payload.get("visualize_seconds")),
                perception_latency_ms=perception_latency_ms,
                summary=dict(payload),
                normalized_prediction=normalized.to_dict(),
                raw_prediction=normalized,
            )
            results.append(result)
            self._processed_samples.add(sample_name)
            self._inferred_samples += 1
            self._last_emitted_prediction = sample_name
            self._emit_health("infering", current_sample=sample_name, notes=[status])

        results.sort(key=lambda item: item.sequence_index)
        if not results:
            self._emit_health("idle", notes=["no_new_predictions"])
        return results

    def build_summary(self) -> Dict[str, Any]:
        worker_exit_code = self._worker_exit_code
        if self._stage1_process is not None:
            worker_exit_code = self._stage1_process.poll()
        return {
            "session_root": str(self.session_root),
            "launch_mode": self.launch_mode,
            "observed_captures": int(len(self._processed_samples) + len(self._capture_registry)),
            "history_skipped_samples": int(self._history_skipped_samples),
            "inferred_samples": int(self._inferred_samples),
            "unresolved_captures": int(len(self._capture_registry)),
            "worker_exit_code": worker_exit_code,
            "worker_failure_class": self._worker_failure_class,
            "worker_root_cause": self._worker_root_cause,
        }

    def unresolved_capture_count(self) -> int:
        return int(len(self._capture_registry))

    def worker_failed(self) -> bool:
        exit_code = self._worker_exit_code
        if self._stage1_process is not None:
            exit_code = self._stage1_process.poll()
        if exit_code is not None and exit_code != 0:
            self._emit_health("dead", notes=[f"worker_exit_code={exit_code}"])
        return exit_code is not None and exit_code != 0

    def close(self) -> None:
        process = self._stage1_process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10.0)
            self._worker_exit_code = process.poll()
            self._stage1_process = None
        if self._worker_log_handle is not None:
            self._worker_log_handle.close()
            self._worker_log_handle = None
        self._write_worker_audit()
        self._emit_health("dead", notes=[f"final_exit_code={self._worker_exit_code}"])
