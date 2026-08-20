from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class CaptureConfig:
    sample_hz: float = 20.0
    window_seconds: float = 2.0
    min_distance_cm: float = 2.0
    max_distance_cm: float = 250.0
    gate_range_cm: float = 8.0
    gate_min_valid_samples: int = 8


@dataclass(slots=True)
class CollectionConfig:
    output_dir: str = "data/raw"
    gesture_seconds: float = 2.0
    settle_seconds: float = 1.0


@dataclass(slots=True)
class SensorConfig:
    driver: str = "mock"
    trigger_pin: int = 23
    echo_pin: int = 24
    max_distance_cm: float = 250.0


@dataclass(slots=True)
class DisplayConfig:
    driver: str = "console"
    width: int = 128
    height: int = 64
    i2c_port: int = 1
    i2c_address: str = "0x3C"


@dataclass(slots=True)
class ModelConfig:
    implementation_path: str = "src/gesture_detection/ml/user_model.py"
    weights_path: str = "artifacts/model_weights.json"


@dataclass(slots=True)
class AppConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_path(cls, path: str | Path) -> "AppConfig":
        raw = json.loads(Path(path).read_text())
        return cls(
            capture=CaptureConfig(**raw.get("capture", {})),
            collection=CollectionConfig(**raw.get("collection", {})),
            sensor=SensorConfig(**raw.get("sensor", {})),
            display=DisplayConfig(**raw.get("display", {})),
            model=ModelConfig(**raw.get("model", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
