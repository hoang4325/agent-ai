from __future__ import annotations

import logging
import queue
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import carla

from .constants import SENSOR_NAMES, assert_sensor_name_set

LOGGER = logging.getLogger(__name__)


@dataclass
class SensorPacket:
    name: str
    frame: int
    timestamp: float
    type_id: str
    measurement: object


@dataclass
class FrameCapture:
    frame: int
    world_snapshot: carla.WorldSnapshot
    packets: Dict[str, SensorPacket]

    @property
    def timestamps(self) -> Dict[str, float]:
        return {name: packet.timestamp for name, packet in self.packets.items()}


class CarlaSynchronousMode:
    def __init__(
        self,
        world: carla.World,
        *,
        fixed_delta_seconds: float,
        traffic_manager: carla.TrafficManager | None = None,
        synchronous_master: bool = True,
    ) -> None:
        self.world = world
        self.fixed_delta_seconds = float(fixed_delta_seconds)
        self.traffic_manager = traffic_manager
        self.synchronous_master = synchronous_master
        self._original_settings: carla.WorldSettings | None = None
        self._tm_original_sync: bool | None = None

    def __enter__(self) -> "CarlaSynchronousMode":
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        if self.traffic_manager is not None:
            self._tm_original_sync = False
            self.traffic_manager.set_synchronous_mode(True)
        LOGGER.info(
            "CARLA synchronous mode enabled (fixed_delta_seconds=%.3f)", self.fixed_delta_seconds
        )
        return self

    def tick(self, timeout: float = 10.0) -> int:
        return int(self.world.tick())

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.traffic_manager is not None:
            self.traffic_manager.set_synchronous_mode(False)
        if self._original_settings is not None:
            self.world.apply_settings(self._original_settings)
        LOGGER.info("CARLA synchronous mode restored")


class FrameCollector:
    def __init__(
        self,
        sensor_actors: Mapping[str, carla.Actor],
        *,
        timeout_seconds: float = 2.0,
    ) -> None:
        assert_sensor_name_set(sensor_actors.keys())
        self.sensor_actors = dict(sensor_actors)
        self.timeout_seconds = float(timeout_seconds)
        self._queue: "queue.Queue[SensorPacket]" = queue.Queue()
        self._pending: MutableMapping[int, Dict[str, SensorPacket]] = defaultdict(dict)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        for name, actor in self.sensor_actors.items():
            actor.listen(self._make_callback(name, actor.type_id))
        self._started = True
        LOGGER.info("FrameCollector listening to %d sensors", len(self.sensor_actors))

    def stop(self) -> None:
        for actor in self.sensor_actors.values():
            try:
                actor.stop()
            except RuntimeError:
                LOGGER.warning("Unable to stop sensor actor id=%s", getattr(actor, "id", "unknown"))
        self._started = False

    def _make_callback(self, name: str, type_id: str):
        def _callback(measurement) -> None:
            packet = SensorPacket(
                name=name,
                frame=int(measurement.frame),
                timestamp=float(measurement.timestamp),
                type_id=type_id,
                measurement=measurement,
            )
            self._queue.put(packet)

        return _callback

    def wait_for_frame(
        self,
        expected_frame: int,
        world_snapshot: carla.WorldSnapshot,
    ) -> FrameCapture:
        if not self._started:
            raise RuntimeError("FrameCollector.start() must be called before wait_for_frame().")

        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            existing = self._pending.get(expected_frame)
            if existing and len(existing) == len(self.sensor_actors):
                packets = self._pending.pop(expected_frame)
                self._drop_stale_frames(expected_frame)
                self._assert_frame_integrity(expected_frame, packets)
                return FrameCapture(frame=expected_frame, world_snapshot=world_snapshot, packets=packets)

            remaining = deadline - time.monotonic()
            try:
                packet = self._queue.get(timeout=max(remaining, 0.001))
            except queue.Empty as exc:
                raise TimeoutError(
                    f"Timed out while waiting for sensor packets of frame {expected_frame}"
                ) from exc

            if packet.frame < expected_frame:
                LOGGER.warning(
                    "Dropping stale packet frame=%d sensor=%s while waiting for frame=%d",
                    packet.frame,
                    packet.name,
                    expected_frame,
                )
                continue

            bucket = self._pending[packet.frame]
            if packet.name in bucket:
                LOGGER.warning(
                    "Duplicate packet frame=%d sensor=%s detected; overwriting previous packet",
                    packet.frame,
                    packet.name,
                )
            bucket[packet.name] = packet

        raise TimeoutError(f"Timed out while assembling synchronized frame {expected_frame}")

    def _drop_stale_frames(self, current_frame: int) -> None:
        stale = [frame for frame in self._pending.keys() if frame < current_frame]
        for frame in stale:
            self._pending.pop(frame, None)

    def _assert_frame_integrity(self, expected_frame: int, packets: Mapping[str, SensorPacket]) -> None:
        actual_names = set(packets.keys())
        expected_names = set(self.sensor_actors.keys())
        if actual_names != expected_names:
            raise ValueError(
                f"Incomplete frame {expected_frame}: expected={sorted(expected_names)} actual={sorted(actual_names)}"
            )
        mismatched = {
            name: packet.frame for name, packet in packets.items() if packet.frame != expected_frame
        }
        if mismatched:
            raise ValueError(f"Frame mismatch for frame {expected_frame}: {mismatched}")

    def pending_frames(self) -> Dict[int, Dict[str, SensorPacket]]:
        return {frame: dict(packets) for frame, packets in self._pending.items()}
