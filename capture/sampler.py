"""
Continuous distance sampler.

Reads from the HC-SR04 at the configured rate, fills missing / dropped
readings, and yields fixed-length windows suitable for gesture inference
or data collection.

Key design decisions
--------------------
* Dropped readings (None from the sensor) are forward-filled from the
  last valid sample.  If no valid sample has been seen yet they are
  filled with MAX_DISTANCE_CM (i.e. "no hand present").
* A sliding window of WINDOW_SAMPLES is maintained.  A new window is
  made available every WINDOW_SECONDS (non-overlapping) but callers can
  choose to overlap windows if they need higher temporal resolution.
"""

import time
from collections import deque
from typing import Callable

import config
from hardware.sensor import HCSR04


class Sampler:
    """
    Continuously samples distance and produces fixed-length windows.

    Parameters
    ----------
    sensor:         An HCSR04 instance (real or simulated).
    on_window:      Callback invoked with a list[float] whenever a full
                    window of WINDOW_SAMPLES is ready.
    overlap:        Fraction of the window to overlap with the next one
                    (0.0 = no overlap, 0.5 = 50 % overlap).
    """

    def __init__(
        self,
        sensor: HCSR04,
        on_window: Callable[[list[float]], None],
        overlap: float = 0.0,
    ):
        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in [0, 1)")
        self.sensor = sensor
        self.on_window = on_window
        self.overlap = overlap

        self._buffer: deque[float] = deque(maxlen=config.WINDOW_SAMPLES)
        self._last_valid: float = config.MAX_DISTANCE_CM

        step_fraction = 1.0 - overlap
        self._step = max(1, int(config.WINDOW_SAMPLES * step_fraction))
        self._samples_since_last_window = 0
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Block and sample continuously until ``stop()`` is called.

        The target inter-sample delay is 1 / SAMPLE_RATE_HZ.  Timing
        accuracy is best-effort; RPi GPIO + Python will have jitter.
        """
        period = 1.0 / config.SAMPLE_RATE_HZ
        self._running = True

        try:
            while self._running:
                t0 = time.monotonic()
                self._tick()
                elapsed = time.monotonic() - t0
                sleep_for = period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            self.sensor.cleanup()

    def stop(self) -> None:
        """Signal the run loop to exit after the current sample."""
        self._running = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Read one sample, fill if necessary, update buffer."""
        raw = self.sensor.read_cm()
        if raw is None:
            # Forward-fill from last known good value
            sample = self._last_valid
        else:
            sample = raw
            self._last_valid = raw

        self._buffer.append(sample)
        self._samples_since_last_window += 1

        if (
            len(self._buffer) == config.WINDOW_SAMPLES
            and self._samples_since_last_window >= self._step
        ):
            self.on_window(list(self._buffer))
            self._samples_since_last_window = 0
