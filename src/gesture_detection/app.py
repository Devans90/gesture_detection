from __future__ import annotations

import time

from gesture_detection.capture import ActivityGate, CaptureSession, RollingWindow
from gesture_detection.config import AppConfig
from gesture_detection.domain import GestureWindow
from gesture_detection.factories import build_classifier, build_display, build_sensor


class GestureDetectionApp:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._sensor = build_sensor(config)
        self._display = build_display(config)
        self._classifier = build_classifier(config)
        self._capture_session = CaptureSession(
            sensor=self._sensor,
            sample_hz=config.capture.sample_hz,
            max_distance_cm=config.capture.max_distance_cm,
        )
        self._rolling_window = RollingWindow(
            capacity=max(1, int(config.capture.window_seconds * config.capture.sample_hz))
        )
        self._activity_gate = ActivityGate(
            min_valid_samples=config.capture.gate_min_valid_samples,
            minimum_range_cm=config.capture.gate_range_cm,
        )
        self._cooldown_until = 0.0

    def run_forever(self) -> None:
        self._display.show_lines([
            "Gesture detector",
            f"sensor={self._config.sensor.driver}",
            f"display={self._config.display.driver}",
            "Waiting for motion...",
        ])

        try:
            while True:
                reading = self._capture_session.read_once()
                self._rolling_window.append(reading)
                snapshot = self._rolling_window.snapshot()
                self._display.show_lines(self._status_lines(reading, snapshot))

                if time.time() >= self._cooldown_until and self._activity_gate.is_active(snapshot):
                    prediction = self._classifier.predict(
                        GestureWindow(started_at=snapshot[0].timestamp, label=None, readings=snapshot)
                    )
                    self._display.show_lines([
                        f"distance={self._distance_text(reading)}",
                        f"status={reading.status}",
                        f"gesture={prediction.label or 'TODO'}",
                        prediction.status_message[:24],
                    ])
                    self._cooldown_until = time.time() + self._config.capture.window_seconds

                time.sleep(1.0 / self._config.capture.sample_hz)
        except KeyboardInterrupt:
            self._display.show_lines(["Gesture detector", "Stopped", "", ""])
        finally:
            self._sensor.close()
            self._display.close()

    @staticmethod
    def _distance_text(reading) -> str:
        if reading.distance_cm is None:
            return "--"
        return f"{reading.distance_cm:.1f}cm"

    def _status_lines(self, reading, snapshot) -> list[str]:
        valid_count = sum(1 for item in snapshot if item.is_valid)
        return [
            "Gesture detector",
            f"distance={self._distance_text(reading)}",
            f"status={reading.status}",
            f"valid={valid_count}/{len(snapshot)}",
        ]
