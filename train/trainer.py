"""
Training script with MLflow tracking.

Run from the project root:
    python -m train.trainer

Prerequisites:
    pip install torch mlflow

After training, weights are exported to MODEL_WEIGHTS_PATH as a .npz
file suitable for the numpy inference engine (infer/numpy_model.py).

-----------------------------------------------------------------------
YOUR TASK (in this file)
-----------------------------------------------------------------------
This script is mostly wired up.  The things left for you:
  1. Implement your model in train/model.py.
  2. (Optional) tune hyper-parameters in config.py.
  3. (Optional) extend _export_weights to serialise every layer you add.
-----------------------------------------------------------------------
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from train.dataset import build_loaders
from train.model import GestureCNN

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import mlflow
    import mlflow.pytorch
    _DEPS_AVAILABLE = True
except ImportError as exc:
    _DEPS_AVAILABLE = False
    _IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Weight export
# ---------------------------------------------------------------------------

def _export_weights(model: "GestureCNN", path: str) -> None:
    """
    Serialise every named parameter to a .npz file.

    The numpy inference engine loads these by name, so the keys here
    must match what numpy_model.py expects.

    TODO
    ----
    Extend this function if you rename or add layers.  For each layer
    add a line like:
        arrays["conv1.weight"] = model.conv1.weight.detach().cpu().numpy()
        arrays["conv1.bias"]   = model.conv1.bias.detach().cpu().numpy()
    """
    import numpy as np

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {
        name: param.detach().cpu().numpy()
        for name, param in model.named_parameters()
    }
    np.savez(path, **arrays)
    print(f"Weights exported → {path}  ({len(arrays)} arrays)")


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def _eval_epoch(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _DEPS_AVAILABLE:
        print(f"ERROR: missing dependency — {_IMPORT_ERROR}")
        print("Install with:  pip install torch mlflow")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_loaders()

    model = GestureCNN().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "window_samples": config.WINDOW_SAMPLES,
            "model_params": model.count_parameters(),
        })

        best_val_acc = 0.0

        for epoch in range(1, config.EPOCHS + 1):
            train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = _eval_epoch(model, val_loader, criterion, device)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            print(
                f"Epoch {epoch:3d}/{config.EPOCHS}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.3f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "models/best_model.pt")

        mlflow.log_metric("best_val_acc", best_val_acc)
        print(f"\nBest val accuracy: {best_val_acc:.3f}")

    # Load best weights before exporting
    model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
    _export_weights(model, config.MODEL_WEIGHTS_PATH)


if __name__ == "__main__":
    main()
