from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class SensorReading:
    timestamp: float
    distance_cm: Optional[float]
    status: str = "ok"

    @property
    def is_valid(self) -> bool:
        return self.distance_cm is not None and self.status == "ok"


@dataclass(slots=True)
class GestureWindow:
    started_at: float
    label: Optional[str]
    readings: list[SensorReading] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def valid_readings(self) -> list[SensorReading]:
        return [reading for reading in self.readings if reading.is_valid]


@dataclass(slots=True)
class Prediction:
    label: Optional[str]
    confidence: float
    status_message: str = ""
