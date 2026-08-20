"""
deploy/export.py — export trained PyTorch weights to numpy arrays.

Run this once after training (on your laptop / desktop), then copy the
resulting .npz file to the Pi.  The Pi runs forward.py — which imports
only numpy — to perform inference without PyTorch or ONNX installed.

Usage:
    python -m deploy.export --checkpoint checkpoints/best.pt --out deploy/weights.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from model import GestureNet, load_weights


def export_weights(checkpoint: str, out_path: str) -> None:
    """
    Load a trained GestureNet checkpoint and save all weight arrays to .npz.

    The exported keys mirror the PyTorch state_dict names, e.g.:
        conv1.weight, conv1.bias, fc.weight, fc.bias, ...

    These exact key names are used by forward.py to reconstruct the network.
    """
    model = GestureNet()
    load_weights(model, checkpoint)
    model.eval()

    arrays = {name: param.detach().cpu().numpy()
              for name, param in model.state_dict().items()}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    print(f"Exported {len(arrays)} weight arrays → {out_path}")
    for name, arr in arrays.items():
        print(f"  {name:40s}  {arr.shape}  {arr.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export weights to numpy .npz")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--out", default="deploy/weights.npz")
    args = parser.parse_args()
    export_weights(args.checkpoint, args.out)
