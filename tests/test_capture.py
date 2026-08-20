"""
Tests for the capture pipeline (no hardware required).

Run with:
    python -m pytest tests/
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from capture.sampler import Sampler
from hardware.sensor import HCSR04


# ---------------------------------------------------------------------------
# HCSR04 simulated sensor
# ---------------------------------------------------------------------------

class TestHCSR04:
    def test_simulated_reading_in_range(self):
        sensor = HCSR04(simulate=True)
        for _ in range(20):
            val = sensor.read_cm()
            assert val is not None
            assert config.MIN_DISTANCE_CM <= val <= config.MAX_DISTANCE_CM

    def test_clamp_rejects_below_min(self):
        result = HCSR04._clamp(0.5)
        assert result is None

    def test_clamp_rejects_above_max(self):
        result = HCSR04._clamp(999.0)
        assert result is None

    def test_clamp_accepts_valid(self):
        result = HCSR04._clamp(20.0)
        assert result == 20.0


# ---------------------------------------------------------------------------
# Sampler window collection
# ---------------------------------------------------------------------------

class TestSampler:
    def test_window_correct_length(self):
        """A full window callback must receive exactly WINDOW_SAMPLES samples."""
        received = []
        sensor = HCSR04(simulate=True)

        def collect(window):
            received.append(window)
            sampler.stop()  # stop after first window

        sampler = Sampler(sensor=sensor, on_window=collect, overlap=0.0)
        sampler.run()

        assert len(received) == 1
        assert len(received[0]) == config.WINDOW_SAMPLES

    def test_window_values_in_range(self):
        """All values in the window must be within sensor range."""
        received = []
        sensor = HCSR04(simulate=True)

        def collect(window):
            received.append(window)
            sampler.stop()

        sampler = Sampler(sensor=sensor, on_window=collect, overlap=0.0)
        sampler.run()

        for v in received[0]:
            assert config.MIN_DISTANCE_CM <= v <= config.MAX_DISTANCE_CM

    def test_forward_fill_on_none(self):
        """
        Dropped readings (sensor returns None) should be forward-filled.
        The buffer must still be fully populated.
        """

        class AllNoneSensor:
            def read_cm(self):
                return None
            def cleanup(self):
                pass

        received = []

        def collect(window):
            received.append(window)
            sampler.stop()

        sampler = Sampler(sensor=AllNoneSensor(), on_window=collect, overlap=0.0)
        sampler.run()

        assert len(received[0]) == config.WINDOW_SAMPLES
        # All values should equal the fill value (MAX_DISTANCE_CM)
        assert all(v == config.MAX_DISTANCE_CM for v in received[0])
