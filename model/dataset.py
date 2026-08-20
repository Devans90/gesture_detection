"""
model/dataset.py
----------------
PyTorch Dataset for the collected gesture examples.

Loads ``.npy`` files from ``data/raw/`` (or any compatible directory) and
returns ``(window_tensor, label_index)`` pairs.

Two split strategies are provided:
* **Random split** — baseline; easy to overfit to one person.
* **Session hold-out** — recommended; keeps entire sessions in either train
  or val, so the model must generalise across different recording conditions.

Usage
~~~~~
::

    from model.dataset import GestureDataset, session_holdout_split

    full = GestureDataset("data/raw")
    train_ds, val_ds = session_holdout_split(full, holdout_sessions=["session_20240101_120000"])
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


# ── Label registry ────────────────────────────────────────────────────────────

# Add your gesture labels here in the order you want them indexed.
# The index becomes the class integer fed to the loss function.
GESTURE_LABELS: list[str] = [
    "swipe_left",
    "swipe_right",
    "wave",
    "push",
    "idle",         # ← include idle so the model can learn to say "nothing"
]

LABEL_TO_IDX: dict[str, int] = {lbl: i for i, lbl in enumerate(GESTURE_LABELS)}
IDX_TO_LABEL: dict[int, str] = {i: lbl for lbl, i in LABEL_TO_IDX.items()}
NUM_CLASSES: int = len(GESTURE_LABELS)


# ── Dataset ───────────────────────────────────────────────────────────────────

class GestureDataset(Dataset):
    """
    Loads all ``.npy`` windows found under ``data_dir`` and maps them to
    integer class labels via ``LABEL_TO_IDX``.

    Parameters
    ----------
    data_dir:
        Root directory, e.g. ``"data/raw"``.
    transform:
        Optional callable applied to the raw numpy array *before* conversion
        to a tensor.  Use this for normalisation.

    Each item is a tuple ``(tensor, label_idx)`` where
    ``tensor`` has shape ``(1, WINDOW_SIZE)`` — the channel-first format
    expected by a 1-D convolutional layer.
    """

    def __init__(self, data_dir: str = "data/raw", transform=None) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self._samples: list[tuple[Path, int, str]] = []  # (path, label_idx, session_id)

        self._scan()

    def _scan(self) -> None:
        """Walk data_dir and collect all .npy paths with known labels."""
        for session_dir in sorted(self.data_dir.iterdir()):
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name
            meta_path = session_dir / "metadata.json"
            label: Optional[str] = None

            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                label = meta.get("label")

            for npy_path in sorted(session_dir.glob("*.npy")):
                # Infer label from filename if not in metadata
                inferred = label or npy_path.stem.rsplit("_", 1)[0]
                if inferred not in LABEL_TO_IDX:
                    continue  # skip unknown gestures
                self._samples.append((npy_path, LABEL_TO_IDX[inferred], session_id))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label_idx, _ = self._samples[idx]
        window: np.ndarray = np.load(path).astype(np.float32)

        if self.transform is not None:
            window = self.transform(window)

        # Shape: (1, WINDOW_SIZE) — channel-first for Conv1d
        tensor = torch.from_numpy(window).unsqueeze(0)
        return tensor, label_idx

    def session_ids(self) -> list[str]:
        """Return a deduplicated list of all session IDs in this dataset."""
        seen: list[str] = []
        for _, _, sid in self._samples:
            if sid not in seen:
                seen.append(sid)
        return seen


# ── Split helpers ─────────────────────────────────────────────────────────────

def session_holdout_split(
    dataset: GestureDataset,
    holdout_sessions: list[str],
) -> tuple[Subset, Subset]:
    """
    Split a ``GestureDataset`` by session ID rather than random sampling.

    Parameters
    ----------
    dataset:
        The full dataset.
    holdout_sessions:
        Session IDs to use as the validation set.

    Returns
    -------
    (train_subset, val_subset)
    """
    holdout_set = set(holdout_sessions)
    train_indices, val_indices = [], []

    for i, (_, _, sid) in enumerate(dataset._samples):
        if sid in holdout_set:
            val_indices.append(i)
        else:
            train_indices.append(i)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# ── Normalisation transform ───────────────────────────────────────────────────

class NormaliseWindow:
    """
    Simple per-window min-max normaliser.

    Scales each window to [0, 1] based on the sensor's physical range.
    Pass as ``transform`` to ``GestureDataset``.

    Parameters
    ----------
    min_cm, max_cm:
        Physical range of the sensor in cm.
    """

    def __init__(self, min_cm: float = 2.0, max_cm: float = 400.0) -> None:
        self.min_cm = min_cm
        self.max_cm = max_cm

    def __call__(self, window: np.ndarray) -> np.ndarray:
        return (window - self.min_cm) / (self.max_cm - self.min_cm)
