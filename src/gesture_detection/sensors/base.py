from __future__ import annotations

from abc import ABC, abstractmethod

from gesture_detection.domain import SensorReading


class DistanceSensor(ABC):
    @abstractmethod
    def read(self) -> SensorReading:
        raise NotImplementedError

    def close(self) -> None:
        """Release hardware resources if needed."""
