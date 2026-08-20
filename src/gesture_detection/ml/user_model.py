from __future__ import annotations

from gesture_detection.domain import GestureWindow, Prediction
from gesture_detection.ml.base import GestureClassifier


class UserImplementedGestureClassifier(GestureClassifier):
    """Implement your NumPy-only inference path here once you have trained a model."""

    def __init__(self, weights_path: str) -> None:
        self.weights_path = weights_path

    def predict(self, window: GestureWindow) -> Prediction:
        raise NotImplementedError(
            "Implement the forward pass in UserImplementedGestureClassifier.predict() after you export weights."
        )
