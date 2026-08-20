from __future__ import annotations

from gesture_detection.config import AppConfig
from gesture_detection.display.base import StatusDisplay
from gesture_detection.display.console import ConsoleDisplay
from gesture_detection.display.oled import SSD1306Display
from gesture_detection.ml.base import GestureClassifier
from gesture_detection.ml.placeholders import PromptingGestureClassifier
from gesture_detection.sensors.base import DistanceSensor
from gesture_detection.sensors.hcsr04 import HCSR04Sensor
from gesture_detection.sensors.mock import MockDistanceSensor


def build_sensor(config: AppConfig) -> DistanceSensor:
    if config.sensor.driver == "mock":
        return MockDistanceSensor()
    if config.sensor.driver == "hcsr04":
        return HCSR04Sensor(
            trigger_pin=config.sensor.trigger_pin,
            echo_pin=config.sensor.echo_pin,
            max_distance_cm=config.sensor.max_distance_cm,
        )
    raise ValueError(f"Unsupported sensor driver: {config.sensor.driver}")


def build_display(config: AppConfig) -> StatusDisplay:
    if config.display.driver == "console":
        return ConsoleDisplay()
    if config.display.driver == "ssd1306":
        return SSD1306Display(
            port=config.display.i2c_port,
            address=int(config.display.i2c_address, 16),
            width=config.display.width,
            height=config.display.height,
        )
    raise ValueError(f"Unsupported display driver: {config.display.driver}")


def build_classifier(config: AppConfig) -> GestureClassifier:
    return PromptingGestureClassifier(config.model.implementation_path)
