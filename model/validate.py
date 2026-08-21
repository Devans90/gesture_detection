"""
model/validate.py
-----------------
Session-hold-out evaluation scaffold.

This script loads a trained checkpoint, runs it on held-out sessions, and
reports per-class accuracy plus a confusion matrix.  It also checks the
model's "idle rejection" — how often it fires on the ``idle`` class.

Run::

    python -m model.validate \\
        --checkpoint runs/<run_id>/best.pt \\
        --holdout-sessions session_20240101_120000

Key ideas
~~~~~~~~~
* **Session hold-out** is the honest metric.  If val accuracy came from
  random splitting, the model may have memorised one person's timing.
* **Idle rejection** matters because the deployed model sees mostly idle
  input.  A model that fires ``swipe_left`` during idle is worse than one
  with lower gesture accuracy but correct idle suppression.
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from model.dataset import GestureDataset, NormaliseWindow, session_holdout_split, IDX_TO_LABEL, NUM_CLASSES
from model.network import build_model
from torch.utils.data import DataLoader


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained gesture model")
    p.add_argument("--checkpoint", required=True, help="Path to best.pt")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--holdout-sessions", nargs="+", required=True, metavar="SID")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def evaluate(
    checkpoint: str,
    data_dir: str = "data/raw",
    holdout_sessions: Optional[list[str]] = None,
    batch_size: int = 32,
) -> dict:
    """
    Load model and evaluate on held-out sessions.

    Returns a dict with keys: ``accuracy``, ``per_class``, ``confusion``.
    """
    device = torch.device("cpu")  # validation runs on CPU / Pi
    model = build_model()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    transform = NormaliseWindow()
    full_ds = GestureDataset(data_dir=data_dir, transform=transform)

    if holdout_sessions:
        _, val_ds = session_holdout_split(full_ds, holdout_sessions=holdout_sessions)
    else:
        val_ds = full_ds

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1).numpy()
            for true, pred in zip(y.numpy(), preds):
                confusion[true][pred] += 1

    per_class: dict[str, float] = {}
    for idx in range(NUM_CLASSES):
        total = confusion[idx].sum()
        per_class[IDX_TO_LABEL[idx]] = float(confusion[idx][idx] / total) if total else float("nan")

    total_correct = int(np.diag(confusion).sum())
    total_samples = int(confusion.sum())
    accuracy = total_correct / total_samples if total_samples else float("nan")

    return {"accuracy": accuracy, "per_class": per_class, "confusion": confusion}


def main() -> None:
    args = _parse_args()
    results = evaluate(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        holdout_sessions=args.holdout_sessions,
        batch_size=args.batch_size,
    )

    print(f"\nOverall accuracy: {results['accuracy']:.3f}")
    print("\nPer-class accuracy:")
    for label, acc in results["per_class"].items():
        print(f"  {label:<20s}  {acc:.3f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    labels = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    header = "            " + "  ".join(f"{l[:8]:>8}" for l in labels)
    print(header)
    for i, row in enumerate(results["confusion"]):
        row_str = "  ".join(f"{v:>8d}" for v in row)
        print(f"  {labels[i]:<12s}  {row_str}")


if __name__ == "__main__":
    main()
