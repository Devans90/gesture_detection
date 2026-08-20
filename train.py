"""
train.py — MLflow-tracked training loop for GestureNet.

What this file already does
----------------------------
- Loads and preprocesses data via dataset.py
- Sets up MLflow experiment logging
- Runs a standard epoch loop with train and validation phases
- Saves the best checkpoint and logs it as an MLflow artifact
- Prints a classification report at the end

What you still need to implement (marked TODO)
-----------------------------------------------
- choose_optimizer()   — define your optimiser (SGD / Adam / etc.)
- choose_loss()        — define your loss function (CrossEntropyLoss, etc.)
- choose_scheduler()   — optional LR schedule

Usage:
    python train.py
    python train.py --epochs 50 --batch-size 16
    mlflow ui   # open browser to http://localhost:5000 to view runs
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import dataset
from model import GestureNet, count_parameters, save_weights

# ---------------------------------------------------------------------------
# Defaults — override via CLI
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS: int = 40
DEFAULT_BATCH: int = 16
DEFAULT_LR: float = 1e-3
CHECKPOINT_DIR: Path = Path("checkpoints")
MLFLOW_EXPERIMENT: str = "gesture_detection"


# ---------------------------------------------------------------------------
# You implement these  ← TODO
# ---------------------------------------------------------------------------

def choose_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Return the optimiser to use during training.

    Suggestions: Adam, SGD with momentum, AdamW.

    TODO: replace the NotImplementedError with your choice, e.g.:
        return torch.optim.Adam(model.parameters(), lr=lr)
    """
    raise NotImplementedError(
        "choose_optimizer() is not implemented.\n"
        "Open train.py and return your preferred optimiser."
    )


def choose_loss() -> nn.Module:
    """
    Return the loss criterion.

    For multi-class classification CrossEntropyLoss is the standard choice.

    TODO: replace the NotImplementedError with your choice, e.g.:
        return nn.CrossEntropyLoss()
    """
    raise NotImplementedError(
        "choose_loss() is not implemented.\n"
        "Open train.py and return your preferred loss function."
    )


def choose_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Optionally return an LR scheduler, or return None to skip scheduling.

    Suggestions: CosineAnnealingLR, StepLR, ReduceLROnPlateau.

    TODO: return a scheduler or None, e.g.:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    """
    raise NotImplementedError(
        "choose_scheduler() is not implemented.\n"
        "Open train.py and return a scheduler or None."
    )


# ---------------------------------------------------------------------------
# Training loop (complete — do not need to edit)
# ---------------------------------------------------------------------------

def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tx = torch.from_numpy(X)
    if tx.ndim == 2:
        tx = tx.unsqueeze(1)  # (N, WINDOW_SAMPLES) → (N, 1, WINDOW_SAMPLES)
    ty = torch.from_numpy(y)
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle)


def train(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # --- Data ---
    print("Loading data…")
    X_train, y_train, X_val, y_val = dataset.build_datasets()
    train_loader = _to_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _to_loader(X_val, y_val, batch_size, shuffle=False)
    print(f"  Train: {len(X_train)} samples   Val: {len(X_val)} samples")

    # --- Model ---
    model = GestureNet()
    print(f"  Parameters: {count_parameters(model):,}")

    # --- Optimisation objects ---
    optimizer = choose_optimizer(model, lr)
    criterion = choose_loss()
    scheduler = choose_scheduler(optimizer, epochs)

    # --- MLflow ---
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "parameters": count_parameters(model),
        })

        best_val_acc = 0.0
        best_ckpt = CHECKPOINT_DIR / "best.pt"

        for epoch in range(1, epochs + 1):
            t0 = time.monotonic()

            # ---- train phase ----
            model.train()
            train_loss = 0.0
            train_correct = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
                train_correct += (logits.argmax(1) == yb).sum().item()

            if scheduler is not None:
                scheduler.step()

            # ---- validation phase ----
            model.eval()
            val_loss = 0.0
            val_correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item() * len(xb)
                    val_correct += (logits.argmax(1) == yb).sum().item()

            t_train = len(X_train)
            t_val = len(X_val)
            tr_loss = train_loss / t_train
            tr_acc = train_correct / t_train
            vl_loss = val_loss / t_val
            vl_acc = val_correct / t_val
            elapsed = time.monotonic() - t0

            mlflow.log_metrics(
                {"train_loss": tr_loss, "train_acc": tr_acc,
                 "val_loss": vl_loss, "val_acc": vl_acc},
                step=epoch,
            )
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}  "
                f"val_loss={vl_loss:.4f} val_acc={vl_acc:.3f}  "
                f"({elapsed:.1f}s)"
            )

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                save_weights(model, str(best_ckpt))
                print(f"  ↳ new best val_acc={best_val_acc:.3f}, saved {best_ckpt}")

        mlflow.log_artifact(str(best_ckpt))
        mlflow.log_metric("best_val_acc", best_val_acc)

    print(f"\nTraining complete.  Best val_acc={best_val_acc:.3f}")
    print(f"Checkpoint: {best_ckpt}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GestureNet")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr)
