from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gesture_detection.domain import GestureWindow, SensorReading


@dataclass(slots=True)
class DatasetIndex:
    root_dir: Path
    labels: list[str]
    session_ids: list[str]
    samples: list[Path]


def build_dataset_index(root_dir: str | Path) -> DatasetIndex:
    """Index samples stored as <root>/<label>/<session>/sample_XXX.json."""
    root = Path(root_dir)
    labels: set[str] = set()
    session_ids: set[str] = set()
    samples = sorted(root.glob("*/*/sample_*.json"))

    for sample in samples:
        labels.add(sample.parent.parent.name)
        session_ids.add(sample.parent.name)

    return DatasetIndex(
        root_dir=root,
        labels=sorted(labels),
        session_ids=sorted(session_ids),
        samples=samples,
    )


def load_window(path: str | Path) -> GestureWindow:
    payload = json.loads(Path(path).read_text())
    readings = [SensorReading(**reading) for reading in payload["readings"]]
    return GestureWindow(
        started_at=payload["started_at"],
        label=payload.get("label"),
        readings=readings,
        notes=payload.get("notes", {}),
    )
