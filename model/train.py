"""
model/train.py
--------------
Training loop scaffold with MLflow experiment tracking.

Run::

    python -m model.train \\
        --data-dir data/raw \\
        --holdout-sessions session_20240101_120000 \\
        --epochs 30 \\
        --lr 1e-3

What this script handles (scaffolding — nothing left to implement):
    * Dataset loading + session-hold-out split
    * DataLoader construction
    * Optimiser + cross-entropy loss
    * Per-epoch train / val loop with loss and accuracy logging
    * MLflow run creation and metric logging
    * Saving the best checkpoint to ``runs/<run_id>/best.pt``

What YOU need to implement first:
    * ``GestureNet`` in ``model/network.py``
    * ``_motion_detected`` in ``capture/buffer.py`` (so you can collect data)
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False
    print("[train] mlflow not found — metrics will only be printed.")

from model.dataset import GestureDataset, NormaliseWindow, session_holdout_split
from model.network import build_model


# ── CLI args ──────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the gesture CNN")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--holdout-sessions", nargs="*", default=[], metavar="SID",
                   help="Session IDs to hold out for validation")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--experiment", default="gesture_detection")
    return p.parse_args()


# ── Training helpers ──────────────────────────────────────────────────────────

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    """Run one epoch; return (mean_loss, accuracy)."""
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += len(y)

    return total_loss / n, correct / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    transform = NormaliseWindow()
    full_ds = GestureDataset(data_dir=args.data_dir, transform=transform)

    if args.holdout_sessions:
        train_ds, val_ds = session_holdout_split(full_ds, holdout_sessions=args.holdout_sessions)
    else:
        # Fallback: 80/20 random split — acceptable for prototyping only
        n_val = max(1, int(0.2 * len(full_ds)))
        n_train = len(full_ds) - n_val
        from torch.utils.data import random_split
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        print("[train] WARNING: using random split — use --holdout-sessions for honest evaluation.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ── MLflow ────────────────────────────────────────────────────────────────
    if _MLFLOW:
        mlflow.set_experiment(args.experiment)
        run = mlflow.start_run()
        mlflow.log_params(vars(args))
        run_id = run.info.run_id
    else:
        run_id = "local"

    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_ckpt_path = output_dir / "best.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, optimiser, criterion, device, train=True)
        val_loss, val_acc = _run_epoch(model, val_loader, optimiser, criterion, device, train=False)

        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        if _MLFLOW:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  ↑ new best val_acc={best_val_acc:.3f}  saved → {best_ckpt_path}")

    if _MLFLOW:
        mlflow.log_artifact(str(best_ckpt_path))
        mlflow.end_run()

    print(f"\n[train] Done.  Best val_acc={best_val_acc:.3f}  checkpoint={best_ckpt_path}")


if __name__ == "__main__":
    main()
