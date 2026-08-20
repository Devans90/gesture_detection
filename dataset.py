"""
dataset.py — load, preprocess and split labelled gesture data.

All CSV files under data/raw/ are read and combined into numpy arrays
ready for training.

Key decisions left open for you to implement (marked TODO):
  - normalisation strategy
  - augmentation strategy
  - window cropping / alignment
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from collect import DATA_DIR, LABELS, WINDOW_SAMPLES

# Map label string → integer class index
LABEL_TO_IDX: Dict[str, int] = {lbl: i for i, lbl in enumerate(LABELS)}
IDX_TO_LABEL: Dict[int, str] = {i: lbl for lbl, i in LABEL_TO_IDX.items()}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Read all CSV files from DATA_DIR.

    Returns
    -------
    X : float32 array, shape (N, WINDOW_SAMPLES)
        Raw distance time-series, one row per example.
    y : int64 array, shape (N,)
        Class indices.
    session_ids : list of str, length N
        Stem of the source file for each example (used for session-based splits).
    """
    X_rows: List[List[float]] = []
    y_rows: List[int] = []
    session_ids: List[str] = []

    csv_files = sorted(Path(DATA_DIR).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}. Run collect.py first.")

    for path in csv_files:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"]
                if label not in LABEL_TO_IDX:
                    continue
                features = [float(row[f"d{i}"]) for i in range(WINDOW_SAMPLES)]
                X_rows.append(features)
                y_rows.append(LABEL_TO_IDX[label])
                session_ids.append(path.stem)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int64)
    return X, y, session_ids


# ---------------------------------------------------------------------------
# Preprocessing  ← YOU IMPLEMENT THIS
# ---------------------------------------------------------------------------

def preprocess(X: np.ndarray) -> np.ndarray:
    """
    Clean and normalise the raw distance time-series.

    Receives X with shape (N, WINDOW_SAMPLES).  Should return an array of
    the same shape (or a transformed shape if you reshape/expand dims).

    Ideas to consider:
      - Z-score normalisation per sample (subtract mean, divide by std)
      - Min-max scaling to [0, 1]
      - Centering relative to the first reading (removes absolute distance bias)
      - Smoothing / detrending
      - Expanding to (N, 1, WINDOW_SAMPLES) for 1-D CNN input

    TODO: implement your chosen normalisation strategy here.
    """
    raise NotImplementedError(
        "preprocess() is not yet implemented.\n"
        "Edit dataset.py and fill in your normalisation strategy."
    )


# ---------------------------------------------------------------------------
# Augmentation  ← YOU IMPLEMENT THIS
# ---------------------------------------------------------------------------

def augment(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation to the training set to improve generalisation.

    Receives X (N, ...) and y (N,).  Returns augmented (X_aug, y_aug) which
    may be larger than the input.

    Ideas to consider:
      - Time-shift: roll the window left/right by a few samples
      - Amplitude jitter: multiply readings by (1 + small_noise)
      - Gaussian noise injection
      - Mixup between same-class examples

    TODO: implement your chosen augmentation strategy here.
    """
    raise NotImplementedError(
        "augment() is not yet implemented.\n"
        "Edit dataset.py and fill in your augmentation strategy."
    )


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def session_split(
    X: np.ndarray,
    y: np.ndarray,
    session_ids: List[str],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hold out whole recording sessions for validation.

    Splitting by session (rather than by individual sample) prevents the model
    from memorising the exact timing of a single person's recording run.

    Returns (X_train, y_train, X_val, y_val).
    """
    rng = np.random.default_rng(seed)
    unique_sessions = list(dict.fromkeys(session_ids))  # preserve order
    rng.shuffle(unique_sessions)

    n_val = max(1, int(len(unique_sessions) * val_fraction))
    val_sessions = set(unique_sessions[:n_val])

    train_mask = np.array([s not in val_sessions for s in session_ids])
    val_mask = ~train_mask

    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def build_datasets(val_fraction: float = 0.2, augment_train: bool = True):
    """
    Full pipeline: load → preprocess → split → (optionally) augment.

    Returns (X_train, y_train, X_val, y_val) as float32/int64 numpy arrays.
    """
    X_raw, y, session_ids = load_raw()
    X = preprocess(X_raw)
    X_train, y_train, X_val, y_val = session_split(X, y, session_ids, val_fraction)
    if augment_train:
        X_train, y_train = augment(X_train, y_train)
    return X_train, y_train, X_val, y_val
