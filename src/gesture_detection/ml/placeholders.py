from __future__ import annotations

from gesture_detection.domain import GestureWindow, Prediction
from gesture_detection.ml.base import GestureClassifier


class PromptingGestureClassifier(GestureClassifier):
    def __init__(self, implementation_path: str) -> None:
        self._implementation_path = implementation_path

    def predict(self, window: GestureWindow) -> Prediction:
        _ = window
        return Prediction(
            label=None,
            confidence=0.0,
            status_message=f"Implement classifier in {self._implementation_path}",
        )
