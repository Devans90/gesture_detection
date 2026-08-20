from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from gesture_detection.capture import CaptureSession
from gesture_detection.storage import write_gesture_window


@dataclass(slots=True)
class CollectorResult:
    session_id: str
    saved_paths: list[Path]


class DataCollector:
    def __init__(self, capture_session: CaptureSession, output_dir: str, settle_seconds: float) -> None:
        self._capture_session = capture_session
        self._output_dir = output_dir
        self._settle_seconds = settle_seconds

    def collect(self, label: str, samples: int, gesture_seconds: float) -> CollectorResult:
        session_id = uuid4().hex[:12]
        saved_paths: list[Path] = []

        for sample_index in range(samples):
            print(f"Prepare gesture '{label}' for sample {sample_index + 1}/{samples}...")
            time.sleep(self._settle_seconds)
            window = self._capture_session.collect_window(seconds=gesture_seconds, label=label)
            saved_paths.append(
                write_gesture_window(self._output_dir, session_id, sample_index, window)
            )
            print(f"Saved {saved_paths[-1]}")

        return CollectorResult(session_id=session_id, saved_paths=saved_paths)
