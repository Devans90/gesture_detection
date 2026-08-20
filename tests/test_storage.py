from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gesture_detection.domain import GestureWindow, SensorReading
from gesture_detection.storage import write_gesture_window
from gesture_detection.training.dataset import build_dataset_index, load_window


class StorageTests(unittest.TestCase):
    def test_write_and_load_gesture_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            window = GestureWindow(
                started_at=1.23,
                label="swipe",
                readings=[SensorReading(timestamp=1.23, distance_cm=18.0)],
                notes={"operator": "test"},
            )
            output = write_gesture_window(temp_dir, session_id="session-a", sample_index=0, window=window)
            payload = json.loads(Path(output).read_text())

            self.assertEqual(payload["label"], "swipe")

            loaded = load_window(output)
            self.assertEqual(loaded.label, "swipe")
            self.assertEqual(len(loaded.readings), 1)

    def test_build_dataset_index_collects_labels_and_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            window = GestureWindow(started_at=1.23, label="tap", readings=[])
            write_gesture_window(temp_dir, session_id="session-a", sample_index=0, window=window)
            dataset = build_dataset_index(temp_dir)

            self.assertEqual(dataset.labels, ["tap"])
            self.assertEqual(dataset.session_ids, ["session-a"])
            self.assertEqual(len(dataset.samples), 1)


if __name__ == "__main__":
    unittest.main()
