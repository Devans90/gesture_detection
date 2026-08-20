from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from gesture_detection.domain import GestureWindow


def write_gesture_window(root_dir: str | Path, session_id: str, sample_index: int, window: GestureWindow) -> Path:
    label = window.label or "unlabelled"
    output_dir = Path(root_dir) / label / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sample_{sample_index:03d}.json"
    output_path.write_text(json.dumps(asdict(window), indent=2) + "\n")
    return output_path
