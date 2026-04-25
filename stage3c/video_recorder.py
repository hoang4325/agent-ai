from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np


def _camera_transform_for_mode(carla_module: Any, mode: str) -> tuple[Any, Any]:
    attachment_type = getattr(carla_module.AttachmentType, "Rigid", None)
    if mode == "hood":
        transform = carla_module.Transform(
            carla_module.Location(x=1.8, z=1.6),
            carla_module.Rotation(pitch=0.0),
        )
        return transform, attachment_type
    if mode == "topdown":
        transform = carla_module.Transform(
            carla_module.Location(z=22.0),
            carla_module.Rotation(pitch=-90.0),
        )
        return transform, attachment_type

    spring_arm = getattr(carla_module.AttachmentType, "SpringArmGhost", attachment_type)
    transform = carla_module.Transform(
        carla_module.Location(x=-8.0, z=3.5),
        carla_module.Rotation(pitch=-15.0),
    )
    return transform, spring_arm


class Stage3CVideoRecorder:
    def __init__(
        self,
        *,
        world: Any,
        vehicle: Any,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        fov: float,
        mode: str = "chase",
        overlay: bool = True,
        capture_timeout_s: float = 2.0,
    ) -> None:
        import carla  # type: ignore

        self._carla = carla
        self._world = world
        self._vehicle = vehicle
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fps = max(1.0, float(fps))
        self._width = int(width)
        self._height = int(height)
        self._overlay = bool(overlay)
        self._capture_timeout_s = float(capture_timeout_s)
        self._queue: queue.Queue[Tuple[int, np.ndarray]] = queue.Queue()
        self._buffered_frame: Tuple[int, np.ndarray] | None = None
        self._written_frames = 0

        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(self._width))
        blueprint.set_attribute("image_size_y", str(self._height))
        blueprint.set_attribute("fov", str(float(fov)))
        blueprint.set_attribute("sensor_tick", "0.0")
        transform, attachment_type = _camera_transform_for_mode(carla, str(mode))
        self._sensor = world.spawn_actor(
            blueprint,
            transform,
            attach_to=vehicle,
            attachment_type=attachment_type,
        )
        self._sensor.listen(self._on_image)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._output_path), fourcc, self._fps, (self._width, self._height))
        if not self._writer.isOpened():
            self.close()
            raise RuntimeError(f"Failed to open video writer for {self._output_path}")

    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def written_frames(self) -> int:
        return self._written_frames

    def _on_image(self, image: Any) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        frame_bgr = np.ascontiguousarray(array[:, :, :3])
        self._queue.put((int(image.frame), frame_bgr))

    def _get_frame_for_carla_frame(self, carla_frame: int) -> np.ndarray | None:
        deadline = time.time() + self._capture_timeout_s
        while time.time() < deadline:
            item = self._buffered_frame
            if item is None:
                timeout = max(0.01, deadline - time.time())
                try:
                    item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    break
            self._buffered_frame = None
            frame_id, frame_bgr = item
            if frame_id < int(carla_frame):
                continue
            return frame_bgr
        return None

    def _overlay_lines(self, frame_bgr: np.ndarray, lines: Iterable[str]) -> np.ndarray:
        if not self._overlay:
            return frame_bgr
        lines = [str(line) for line in lines]
        result = frame_bgr.copy()
        band_height = 28 + 24 * len(lines)
        cv2.rectangle(result, (0, 0), (self._width, band_height), (16, 16, 16), thickness=-1)
        y = 28
        for line in lines:
            cv2.putText(
                result,
                str(line),
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
            y += 24
        return result

    def write_frame(self, *, carla_frame: int, overlay_lines: List[str]) -> bool:
        frame_bgr = self._get_frame_for_carla_frame(int(carla_frame))
        if frame_bgr is None:
            return False
        frame_bgr = self._overlay_lines(frame_bgr, overlay_lines)
        self._writer.write(frame_bgr)
        self._written_frames += 1
        return True

    def manifest(self) -> Dict[str, Any]:
        return {
            "output_path": str(self._output_path),
            "fps": float(self._fps),
            "width": int(self._width),
            "height": int(self._height),
            "written_frames": int(self._written_frames),
            "overlay": bool(self._overlay),
        }

    def close(self) -> None:
        sensor = getattr(self, "_sensor", None)
        if sensor is not None:
            try:
                sensor.stop()
            except RuntimeError:
                pass
            try:
                sensor.destroy()
            except RuntimeError:
                pass
            self._sensor = None
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.release()
            self._writer = None
