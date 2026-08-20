from __future__ import annotations

import unittest

from gesture_detection.capture import ActivityGate
from gesture_detection.domain import SensorReading


class ActivityGateTests(unittest.TestCase):
    def test_requires_enough_valid_samples(self) -> None:
        gate = ActivityGate(min_valid_samples=3, minimum_range_cm=4.0)
        readings = [
            SensorReading(timestamp=0.0, distance_cm=20.0),
            SensorReading(timestamp=0.1, distance_cm=22.0),
        ]
        self.assertFalse(gate.is_active(readings))

    def test_detects_motion_from_distance_range(self) -> None:
        gate = ActivityGate(min_valid_samples=3, minimum_range_cm=4.0)
        readings = [
            SensorReading(timestamp=0.0, distance_cm=20.0),
            SensorReading(timestamp=0.1, distance_cm=25.0),
            SensorReading(timestamp=0.2, distance_cm=27.0),
        ]
        self.assertTrue(gate.is_active(readings))


if __name__ == "__main__":
    unittest.main()
