from __future__ import annotations

from abc import ABC, abstractmethod

from gesture_detection.domain import GestureWindow, Prediction


class GestureClassifier(ABC):
    @abstractmethod
    def predict(self, window: GestureWindow) -> Prediction:
        raise NotImplementedError
