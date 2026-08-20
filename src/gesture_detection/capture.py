from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from gesture_detection.domain import GestureWindow, SensorReading
from gesture_detection.sensors.base import DistanceSensor


@dataclass(slots=True)
class ActivityGate:
    min_valid_samples: int
    minimum_range_cm: float

    def is_active(self, readings: Iterable[SensorReading]) -> bool:
        valid = [reading.distance_cm for reading in readings if reading.is_valid]
        if len(valid) < self.min_valid_samples:
            return False
        return max(valid) - min(valid) >= self.minimum_range_cm


class CaptureSession:
    def __init__(self, sensor: DistanceSensor, sample_hz: float, max_distance_cm: float) -> None:
        self._sensor = sensor
        self._sample_hz = sample_hz
        self._interval = 1.0 / sample_hz
        self._max_distance_cm = max_distance_cm

    def read_once(self) -> SensorReading:
        reading = self._sensor.read()
        if reading.distance_cm is not None and reading.distance_cm > self._max_distance_cm:
            return SensorReading(
                timestamp=reading.timestamp,
                distance_cm=None,
                status="out_of_range",
            )
        return reading

    def collect_window(self, seconds: float, label: str | None = None) -> GestureWindow:
        total_samples = max(1, int(seconds * self._sample_hz))
        window = GestureWindow(started_at=time.time(), label=label)

        for _ in range(total_samples):
            loop_started = time.time()
            window.readings.append(self.read_once())
            elapsed = time.time() - loop_started
            remaining = self._interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

        return window


class RollingWindow:
    def __init__(self, capacity: int) -> None:
        self._buffer: deque[SensorReading] = deque(maxlen=capacity)

    def append(self, reading: SensorReading) -> None:
        self._buffer.append(reading)

    def snapshot(self) -> list[SensorReading]:
        return list(self._buffer)
