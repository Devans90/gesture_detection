"""
collect/recorder.py
-------------------
Records labelled gesture examples to disk.

Each example is a 1-D numpy array of shape ``(WINDOW_SIZE,)`` (distances in
cm) plus a string label such as ``"swipe_left"``.

Storage layout
~~~~~~~~~~~~~~
::

    data/
        raw/
            session_<YYYYMMDD_HHMMSS>/
                metadata.json          ← session-level info (participant, date)
                swipe_left_000.npy
                swipe_left_001.npy
                wave_000.npy
                ...

A ``sessions.csv`` manifest (updated after each session) lets the training
code do session-hold-out cross-validation without touching the raw files.

Usage
~~~~~
::

    rec = Recorder(label="swipe_left", data_dir="data/raw")
    rec.start_session(participant="alice")
    rec.record(window_array)   # call once per captured window
    ...
    rec.finish_session()
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR: str = "data/raw"
MANIFEST_FILE: str = "data/sessions.csv"
MANIFEST_COLUMNS: list[str] = ["session_id", "participant", "date", "label", "n_examples"]


# ── Recorder ─────────────────────────────────────────────────────────────────

class Recorder:
    """
    Saves labelled gesture windows to ``<data_dir>/<session_id>/``.

    Parameters
    ----------
    label:
        Gesture class name, e.g. ``"swipe_left"``.
    data_dir:
        Root directory for raw data.
    """

    def __init__(self, label: str, data_dir: str = DEFAULT_DATA_DIR) -> None:
        self.label = label
        self.data_dir = Path(data_dir)
        self._session_dir: Optional[Path] = None
        self._session_id: Optional[str] = None
        self._participant: Optional[str] = None
        self._count: int = 0

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def start_session(self, participant: str = "unknown") -> str:
        """
        Create a timestamped session directory and return the session ID.

        Parameters
        ----------
        participant:
            Name or ID of the person performing gestures.  Used to track
            who contributed which examples (important for session-hold-out).
        """
        self._participant = participant
        self._session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._session_dir = self.data_dir / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0

        metadata = {
            "session_id": self._session_id,
            "participant": participant,
            "label": self.label,
            "date": datetime.now().isoformat(),
        }
        with open(self._session_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Recorder] Session started: {self._session_id}  label={self.label}  participant={participant}")
        return self._session_id

    def record(self, window: np.ndarray) -> Path:
        """
        Save one window to disk.

        Parameters
        ----------
        window:
            1-D float32 numpy array of sensor distances.

        Returns
        -------
        Path
            The path to the saved ``.npy`` file.
        """
        if self._session_dir is None:
            raise RuntimeError("Call start_session() before record().")

        filename = f"{self.label}_{self._count:04d}.npy"
        path = self._session_dir / filename
        np.save(path, window.astype(np.float32))
        self._count += 1
        return path

    def finish_session(self) -> None:
        """
        Finalise the session: update ``sessions.csv`` and print a summary.
        """
        if self._session_id is None:
            return

        self._append_manifest()
        print(
            f"[Recorder] Session finished: {self._session_id}  "
            f"examples={self._count}"
        )

        self._session_dir = None
        self._session_id = None
        self._participant = None
        self._count = 0

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _append_manifest(self) -> None:
        manifest_path = Path(MANIFEST_FILE)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not manifest_path.exists()

        with open(manifest_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "session_id": self._session_id,
                "participant": self._participant,
                "date": datetime.now().date().isoformat(),
                "label": self.label,
                "n_examples": self._count,
            })
