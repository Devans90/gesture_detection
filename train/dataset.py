"""
Dataset loading and pre-processing for gesture classification.

This module provides:
    GestureDataset   – loads all CSVs from data/raw/, applies transforms
    build_loaders    – convenience function that returns train/val DataLoaders

Pre-processing steps applied to every window (already implemented):
    1. Clamp distances to [MIN_DISTANCE_CM, MAX_DISTANCE_CM]
    2. Min-max normalise to [0, 1] using the window's own range
       (so the network sees relative shape, not absolute distance)
    3. Reshape to (1, WINDOW_SAMPLES) for a 1-D conv input

-----------------------------------------------------------------------
YOUR TASK
-----------------------------------------------------------------------
The pre-processing above is intentionally minimal.  You may want to:
    * Add z-score normalisation instead of, or in addition to, min-max.
    * Smooth noisy readings with a moving average before feeding the model.
    * Augment the dataset (time-shift, add noise) to improve generalisation.

Open `_preprocess` and `_augment` (both marked TODO) to implement this.
-----------------------------------------------------------------------
"""

import os
import csv
import random
from typing import Tuple

import numpy as np

import config

# Optional: only imported when build_loaders is called
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Low-level CSV loading
# ---------------------------------------------------------------------------

def load_raw_csv(path: str) -> Tuple[list[str], list[list[float]]]:
    """
    Read a single session CSV.

    Returns
    -------
    labels  : list of string labels, one per row
    windows : list of float lists, each of length WINDOW_SAMPLES
    """
    labels, windows = [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels.append(row["label"])
            window = [float(row[f"d_{i}"]) for i in range(config.WINDOW_SAMPLES)]
            windows.append(window)
    return labels, windows


def load_all_raw(data_dir: str = config.DATA_DIR) -> Tuple[list[str], list[list[float]], list[str]]:
    """
    Load every CSV in `data_dir`.

    Returns
    -------
    labels     : list[str]
    windows    : list[list[float]]
    session_ids: list[str]  — filename stem, used for session holdout splits
    """
    all_labels, all_windows, all_sessions = [], [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv"):
            continue
        session_id = os.path.splitext(fname)[0]
        path = os.path.join(data_dir, fname)
        labels, windows = load_raw_csv(path)
        all_labels.extend(labels)
        all_windows.extend(windows)
        all_sessions.extend([session_id] * len(labels))
    return all_labels, all_windows, all_sessions


# ---------------------------------------------------------------------------
# Pre-processing  (TODO: implement / extend)
# ---------------------------------------------------------------------------

def _clamp(window: list[float]) -> list[float]:
    """Clamp readings to the configured sensor range."""
    return [
        max(config.MIN_DISTANCE_CM, min(config.MAX_DISTANCE_CM, v))
        for v in window
    ]


def _minmax_normalise(window: list[float]) -> list[float]:
    """Scale window values to [0, 1] using the window's own min/max."""
    lo, hi = min(window), max(window)
    span = hi - lo
    if span < 1e-6:
        # Flat window (no hand movement): return zeros
        return [0.0] * len(window)
    return [(v - lo) / span for v in window]


def _preprocess(window: list[float]) -> np.ndarray:
    """
    Convert a raw distance window into a model-ready numpy array.

    Shape of the returned array: (1, WINDOW_SAMPLES)
      – channel-first format expected by the 1-D CNN.

    Currently applies: clamp → min-max normalise → reshape.

    TODO
    ----
    Consider adding:
      * Gaussian / moving-average smoothing to reduce HC-SR04 noise.
      * Z-score normalisation relative to a running baseline.
      * Any feature engineering that might help (derivatives, peak count…).
    """
    window = _clamp(window)
    window = _minmax_normalise(window)
    arr = np.array(window, dtype=np.float32)
    return arr.reshape(1, -1)  # (1, WINDOW_SAMPLES)


def _augment(window: np.ndarray) -> np.ndarray:
    """
    Apply optional data augmentation.

    Parameters
    ----------
    window : np.ndarray of shape (1, WINDOW_SAMPLES)

    Returns
    -------
    Augmented window of the same shape.

    TODO
    ----
    Implement one or more of these to improve generalisation:
      * Time-shift: roll the window left/right by a few samples.
      * Additive Gaussian noise: mimic sensor jitter.
      * Amplitude scaling: simulate different hand heights.
      * Random time-stretch / compress (resample with scipy.signal).

    Return the original `window` unchanged if you don't want augmentation.
    """
    # ---- your implementation here ----
    return window


# ---------------------------------------------------------------------------
# Session-based train/val split
# ---------------------------------------------------------------------------

def session_split(
    labels: list[str],
    windows: list[list[float]],
    session_ids: list[str],
    holdout: int = config.VALIDATION_SESSION_HOLDOUT,
) -> Tuple[list, list, list, list]:
    """
    Hold out `holdout` randomly chosen sessions for validation.

    Returns (train_windows, train_labels, val_windows, val_labels).
    """
    unique_sessions = list(set(session_ids))
    random.shuffle(unique_sessions)
    val_sessions = set(unique_sessions[:holdout])

    train_w, train_l, val_w, val_l = [], [], [], []
    for w, lbl, sid in zip(windows, labels, session_ids):
        if sid in val_sessions:
            val_w.append(w)
            val_l.append(lbl)
        else:
            train_w.append(w)
            train_l.append(lbl)

    return train_w, train_l, val_w, val_l


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class GestureDataset(Dataset):
        """
        PyTorch Dataset for gesture windows.

        Parameters
        ----------
        windows    : list of raw distance windows (list[float])
        labels     : list of string labels
        augment    : whether to apply ``_augment`` during __getitem__
        """

        label_to_idx = {lbl: i for i, lbl in enumerate(config.GESTURE_LABELS)}

        def __init__(
            self,
            windows: list[list[float]],
            labels: list[str],
            augment: bool = False,
        ):
            self.windows = windows
            self.labels = labels
            self.augment = augment

        def __len__(self) -> int:
            return len(self.windows)

        def __getitem__(self, idx: int):
            x = _preprocess(self.windows[idx])
            if self.augment:
                x = _augment(x)
            x = torch.from_numpy(x)
            y = torch.tensor(self.label_to_idx[self.labels[idx]], dtype=torch.long)
            return x, y


def build_loaders(
    data_dir: str = config.DATA_DIR,
    batch_size: int = config.BATCH_SIZE,
    holdout: int = config.VALIDATION_SESSION_HOLDOUT,
) -> Tuple["DataLoader", "DataLoader"]:
    """
    Load data, split by session, and return (train_loader, val_loader).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for build_loaders.")

    labels, windows, session_ids = load_all_raw(data_dir)
    train_w, train_l, val_w, val_l = session_split(labels, windows, session_ids, holdout)

    train_ds = GestureDataset(train_w, train_l, augment=True)
    val_ds = GestureDataset(val_w, val_l, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset  train={len(train_ds)}  val={len(val_ds)}")
    return train_loader, val_loader
