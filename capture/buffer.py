"""
capture/buffer.py
-----------------
Rolling distance buffer and gesture-window segmentation.

A *gesture window* is a fixed-length slice of the distance time-series.
This module provides:

1. ``RollingBuffer`` — accumulates raw sensor readings and fills dropped
   samples with a configurable strategy.
2. ``GestureSegmenter`` — watches the buffer and fires when it decides a
   gesture has occurred.  The detection heuristic is intentionally left as
   a stub: fill in ``_motion_detected`` with your own threshold logic.

Design notes
~~~~~~~~~~~~
* Window length is chosen so that the slowest gesture (say 2 s) fits at
  20 Hz → 40 samples.  Adjust ``WINDOW_SIZE`` to taste.
* Dropped readings (``None``) are filled with the last valid reading
  (forward-fill).  If no valid reading exists yet, the fill value is
  ``MAX_DISTANCE_CM`` (hand absent).
"""

from collections import deque
from typing import Callable, Optional
import numpy as np

from capture.sensor import MAX_DISTANCE_CM

# ── Constants ────────────────────────────────────────────────────────────────

WINDOW_SIZE: int = 40          # samples in one gesture window (2 s @ 20 Hz)
IDLE_FILL_CM: float = MAX_DISTANCE_CM  # value used when no prior reading exists


# ── Rolling buffer ────────────────────────────────────────────────────────────

class RollingBuffer:
    """
    Fixed-length FIFO of distance readings with forward-fill for ``None``.

    Parameters
    ----------
    maxlen:
        Number of samples to keep.  Older samples are discarded automatically.
    idle_fill:
        Value used before any valid reading has arrived.
    """

    def __init__(self, maxlen: int = WINDOW_SIZE, idle_fill: float = IDLE_FILL_CM) -> None:
        self._maxlen = maxlen
        self._idle_fill = idle_fill
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._last_valid: float = idle_fill

    def push(self, distance: Optional[float]) -> None:
        """Add one reading, filling ``None`` with the last valid value."""
        if distance is not None:
            self._last_valid = distance
            self._buf.append(distance)
        else:
            self._buf.append(self._last_valid)

    def full(self) -> bool:
        """Return ``True`` once the buffer contains ``maxlen`` samples."""
        return len(self._buf) == self._maxlen

    def as_array(self) -> np.ndarray:
        """Return the buffer as a 1-D float32 numpy array (oldest → newest)."""
        return np.array(self._buf, dtype=np.float32)

    def reset(self) -> None:
        """Clear the buffer (e.g. between gesture collection sessions)."""
        self._buf.clear()
        self._last_valid = self._idle_fill

    def __len__(self) -> int:
        return len(self._buf)


# ── Gesture segmenter ─────────────────────────────────────────────────────────

class GestureSegmenter:
    """
    Watches a ``RollingBuffer`` and calls ``on_gesture`` when motion is
    detected inside the window.

    The detection heuristic (``_motion_detected``) is a placeholder —
    **implement it yourself**.  A simple starting point: check whether the
    standard deviation of the window exceeds a threshold, meaning the hand
    moved significantly.

    Parameters
    ----------
    buffer:
        The rolling buffer to watch.
    on_gesture:
        Callback that receives the window as a ``numpy.ndarray`` of shape
        ``(WINDOW_SIZE,)`` and a boolean flag indicating whether motion was
        detected.
    cooldown_windows:
        How many windows to skip after firing, to avoid double-triggering.
    """

    def __init__(
        self,
        buffer: RollingBuffer,
        on_gesture: Callable[[np.ndarray], None],
        cooldown_windows: int = 1,
    ) -> None:
        self._buffer = buffer
        self._on_gesture = on_gesture
        self._cooldown = cooldown_windows
        self._skip = 0  # remaining windows to skip

    def update(self) -> None:
        """
        Call once per sample after pushing to the buffer.

        When the buffer is full, checks for motion and either fires
        ``on_gesture`` or silently resets the buffer for the next window.
        """
        if not self._buffer.full():
            return

        window = self._buffer.as_array()
        self._buffer.reset()

        if self._skip > 0:
            self._skip -= 1
            return

        if self._motion_detected(window):
            self._skip = self._cooldown
            self._on_gesture(window)

    # ── TODO: implement ───────────────────────────────────────────────────────

    def _motion_detected(self, window: np.ndarray) -> bool:
        """
        Return ``True`` if ``window`` contains a gesture, ``False`` if idle.

        **This is your first ML-adjacent task.**  Start with a simple
        statistical threshold (e.g. ``window.std() > SOME_THRESHOLD``),
        confirm it reliably distinguishes gesture from idle, then move on to
        the neural network classifier once data collection is working.

        Parameters
        ----------
        window:
            1-D float32 array of shape ``(WINDOW_SIZE,)``, distances in cm.
        """
        raise NotImplementedError(
            "Implement _motion_detected in capture/buffer.py.  "
            "A good starting point: return window.std() > <your_threshold>"
        )
