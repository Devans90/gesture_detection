from __future__ import annotations

import time

from gesture_detection.domain import SensorReading
from gesture_detection.sensors.base import DistanceSensor

try:
    from gpiozero import DistanceSensor as GPIOZeroDistanceSensor
except ImportError:  # pragma: no cover - exercised on Raspberry Pi hardware only.
    GPIOZeroDistanceSensor = None


class HCSR04Sensor(DistanceSensor):
    """Thin wrapper around gpiozero for HC-SR04 captures on Raspberry Pi."""

    def __init__(self, trigger_pin: int, echo_pin: int, max_distance_cm: float) -> None:
        if GPIOZeroDistanceSensor is None:
            raise RuntimeError(
                "gpiozero is not installed. Install it on the Raspberry Pi to enable HC-SR04 capture."
            )

        self._max_distance_cm = max_distance_cm
        self._sensor = GPIOZeroDistanceSensor(
            echo=echo_pin,
            trigger=trigger_pin,
            max_distance=max_distance_cm / 100.0,
            queue_len=3,
        )

    def read(self) -> SensorReading:
        timestamp = time.time()
        distance_cm = round(self._sensor.distance * 100.0, 2)

        if distance_cm <= 0:
            return SensorReading(timestamp=timestamp, distance_cm=None, status="dropped")
        if distance_cm > self._max_distance_cm:
            return SensorReading(timestamp=timestamp, distance_cm=None, status="out_of_range")
        return SensorReading(timestamp=timestamp, distance_cm=distance_cm, status="ok")

    def close(self) -> None:
        self._sensor.close()
