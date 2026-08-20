"""
deploy/predictor.py
-------------------
High-level gesture predictor.

Combines ``NumpyInference`` (numpy-only forward pass) with:
* A normalisation step matching training preprocessing.
* **Idle rejection** — suppresses predictions when the model isn't confident
  enough, so the display doesn't fire constantly.

The idle-rejection threshold is intentionally left as a stub.

Your task
~~~~~~~~~
After implementing ``NumpyInference.predict`` and measuring model outputs on
real idle and gesture inputs, tune ``CONFIDENCE_THRESHOLD`` and optionally
override ``_is_idle`` with a more sophisticated rule.
"""

from typing import Optional

import numpy as np

from capture.buffer import WINDOW_SIZE
from deploy.inference import NumpyInference, softmax
from model.dataset import IDX_TO_LABEL, NUM_CLASSES, LABEL_TO_IDX

# ── Constants ─────────────────────────────────────────────────────────────────

IDLE_LABEL: str = "idle"

# Minimum softmax probability for a prediction to be accepted.
# If the highest class probability is below this, the predictor returns None
# (treated as "nothing happening").
#
# TODO: tune this after you have measured real model outputs on idle input.
CONFIDENCE_THRESHOLD: float = 0.7

# Sensor physical range (must match training normalisation)
_SENSOR_MIN_CM: float = 2.0
_SENSOR_MAX_CM: float = 400.0


# ── Predictor ─────────────────────────────────────────────────────────────────

class Predictor:
    """
    Wraps ``NumpyInference`` with normalisation and idle rejection.

    Parameters
    ----------
    weights_path:
        Path to ``weights.npz`` produced by ``deploy/export.py``.
    confidence_threshold:
        Minimum confidence to accept a prediction.
    """

    def __init__(
        self,
        weights_path: str = "deploy/weights.npz",
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self._inference = NumpyInference(weights_path=weights_path)
        self._threshold = confidence_threshold

    def predict(self, window: np.ndarray) -> Optional[str]:
        """
        Classify a gesture window.

        Parameters
        ----------
        window:
            1-D float32 array of shape ``(WINDOW_SIZE,)``, raw distances in cm.

        Returns
        -------
        str or None
            The predicted gesture label, or ``None`` if confidence is below
            threshold or the model predicts "idle".
        """
        normalised = self._normalise(window)
        logits = self._inference.predict(normalised)
        probs = softmax(logits)

        best_idx: int = int(np.argmax(probs))
        best_prob: float = float(probs[best_idx])
        label = IDX_TO_LABEL.get(best_idx, "unknown")

        if self._is_idle(label, best_prob):
            return None

        return label

    # ── Overridable hooks ─────────────────────────────────────────────────────

    def _normalise(self, window: np.ndarray) -> np.ndarray:
        """Min-max normalise window to [0, 1] matching training preprocessing."""
        return (window - _SENSOR_MIN_CM) / (_SENSOR_MAX_CM - _SENSOR_MIN_CM)

    def _is_idle(self, label: str, confidence: float) -> bool:
        """
        Return ``True`` if this prediction should be suppressed.

        The default rule:
        1. Explicit "idle" class prediction → suppress.
        2. Confidence below threshold → suppress.

        TODO: after evaluating the model, you may want a more nuanced rule,
        e.g. a rolling vote over the last N windows, or a separate idle
        detector based on ``window.std()``.
        """
        if label == IDLE_LABEL:
            return True
        if confidence < self._threshold:
            return True
        return False
