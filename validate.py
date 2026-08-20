"""
validate.py — session-held-out evaluation and rejection analysis.

Runs the trained model against held-out validation sessions and produces:
  - per-class accuracy and confusion matrix
  - rejection analysis: how often the model fires on "idle" motion
  - confidence histogram to help choose a rejection threshold

The validation strategy deliberately holds out complete recording sessions
rather than random samples, matching the real deployment scenario where the
model encounters a new person or recording context.

Usage:
    python validate.py
    python validate.py --checkpoint checkpoints/best.pt --threshold 0.7
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import dataset
from collect import LABELS
from model import GestureNet, load_weights, N_CLASSES

CHECKPOINT_DEFAULT = "checkpoints/best.pt"


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax (used after extracting logits as numpy)."""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    cm = confusion_matrix(y_true, y_pred, N_CLASSES)
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    header = "        " + "  ".join(f"{lbl[:8]:>8}" for lbl in LABELS)
    print(header)
    for i, lbl in enumerate(LABELS):
        row = "  ".join(f"{cm[i, j]:8d}" for j in range(N_CLASSES))
        print(f"  {lbl[:8]:>8}  {row}")

    print("\nPer-class accuracy:")
    for i, lbl in enumerate(LABELS):
        total = cm[i].sum()
        correct = cm[i, i]
        acc = correct / total if total else 0.0
        print(f"  {lbl:<14}  {correct}/{total}  ({acc:.1%})")

    overall = (y_true == y_pred).mean()
    print(f"\nOverall accuracy: {overall:.1%}")


def rejection_analysis(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    idle_idx: int,
) -> None:
    """
    Show how a confidence threshold affects false-positive (idle) rate.

    A prediction is 'rejected' when max confidence < threshold.
    The ideal behaviour:
      - idle examples are rejected (or correctly predicted as idle)
      - gesture examples are accepted at high confidence
    """
    max_conf = probs.max(axis=1)
    accepted = max_conf >= threshold
    rejected = ~accepted

    idle_mask = y_true == idle_idx
    gesture_mask = ~idle_mask

    idle_accepted = (idle_mask & accepted).sum()
    idle_total = idle_mask.sum()
    gesture_rejected = (gesture_mask & rejected).sum()
    gesture_total = gesture_mask.sum()

    print(f"\nRejection analysis  (threshold={threshold:.2f}):")
    print(f"  Idle examples accepted (false positives): "
          f"{idle_accepted}/{idle_total}  ({idle_accepted/max(idle_total,1):.1%})")
    print(f"  Gesture examples rejected (false negatives): "
          f"{gesture_rejected}/{gesture_total}  ({gesture_rejected/max(gesture_total,1):.1%})")

    # Confidence percentiles
    print("\n  Max-confidence percentiles over all examples:")
    for pct in [10, 25, 50, 75, 90]:
        val = np.percentile(max_conf, pct)
        print(f"    p{pct:2d}: {val:.3f}")


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate(checkpoint: str = CHECKPOINT_DEFAULT, threshold: float = 0.7) -> None:
    print(f"Loading checkpoint: {checkpoint}")
    model = GestureNet()
    load_weights(model, checkpoint)
    model.eval()

    print("Loading validation data…")
    X_raw, y, session_ids = dataset.load_raw()
    X = dataset.preprocess(X_raw)
    _, _, X_val, y_val = dataset.session_split(X, y, session_ids)

    if X_val.ndim == 2:
        X_val_t = torch.from_numpy(X_val).unsqueeze(1)
    else:
        X_val_t = torch.from_numpy(X_val)

    print(f"Validation samples: {len(X_val)}")

    with torch.no_grad():
        logits = model(X_val_t).numpy()

    probs = softmax(logits)
    y_pred = probs.argmax(axis=1)

    print_classification_report(y_val, y_pred)

    idle_idx = LABELS.index("idle") if "idle" in LABELS else -1
    if idle_idx >= 0:
        rejection_analysis(y_val, probs, threshold, idle_idx)
    else:
        print("\n(No 'idle' class found — skipping rejection analysis)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate GestureNet")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Confidence threshold for rejection (default 0.7)")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.threshold)
