from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gesture_detection.config import AppConfig


class ConfigTests(unittest.TestCase):
    def test_write_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'config.json'
            config = AppConfig()
            config.sensor.driver = 'mock'
            config.write(path)

            loaded = AppConfig.from_path(path)

            self.assertEqual(loaded.sensor.driver, 'mock')
            self.assertEqual(loaded.capture.sample_hz, config.capture.sample_hz)


if __name__ == '__main__':
    unittest.main()
