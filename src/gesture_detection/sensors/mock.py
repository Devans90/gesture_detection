from __future__ import annotations

import math
import random
import time

from gesture_detection.domain import SensorReading
from gesture_detection.sensors.base import DistanceSensor


class MockDistanceSensor(DistanceSensor):
    """Local-development fallback that emits a noisy wave gesture pattern."""

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self._tick = 0

    def read(self) -> SensorReading:
        self._tick += 1
        timestamp = time.time()

        if self._tick % 37 == 0:
            return SensorReading(timestamp=timestamp, distance_cm=None, status="dropped")

        base = 28.0 + math.sin(self._tick / 4.0) * 10.0
        noise = self._rng.uniform(-1.0, 1.0)
        distance_cm = round(base + noise, 2)
        return SensorReading(timestamp=timestamp, distance_cm=distance_cm, status="ok")
