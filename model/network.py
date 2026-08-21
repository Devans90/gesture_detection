"""
model/network.py
----------------
Neural network definition for gesture classification.

The sensor produces a 1-D time-series of length ``WINDOW_SIZE`` (default 40).
A 1-D CNN is a natural fit: local convolutional filters can learn short-range
patterns (a sharp dip, a plateau) and stack into longer-range features.

This file contains:
* ``GestureNet`` — a PyTorch ``nn.Module`` with ``__init__`` and ``forward``
  **left for you to implement**.
* ``build_model`` — convenience factory used by the training script.

Your task
~~~~~~~~~
1. Design the layers in ``GestureNet.__init__``.
2. Wire them together in ``GestureNet.forward``.

A suggested architecture to get you started (you are free to change it):
    * ``Conv1d(1, 16, kernel_size=5, padding=2)`` + ReLU
    * ``Conv1d(16, 32, kernel_size=5, padding=2)`` + ReLU + MaxPool1d(2)
    * ``Conv1d(32, 64, kernel_size=3, padding=1)`` + ReLU + MaxPool1d(2)
    * Flatten → Linear(64 * 10, 64) → ReLU → Dropout(0.3) → Linear(64, NUM_CLASSES)

But experiment — the dataset is small enough to iterate quickly.
"""

import torch
import torch.nn as nn

from model.dataset import NUM_CLASSES
from capture.buffer import WINDOW_SIZE


class GestureNet(nn.Module):
    """
    1-D CNN gesture classifier.

    Input:  ``(batch, 1, WINDOW_SIZE)`` float32 tensor.
    Output: ``(batch, NUM_CLASSES)`` logit tensor.

    Parameters
    ----------
    num_classes:
        Number of gesture categories.  Defaults to ``NUM_CLASSES`` from
        ``model/dataset.py``.
    window_size:
        Length of the input time-series.  Must match ``WINDOW_SIZE`` used
        during data collection.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, window_size: int = WINDOW_SIZE) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.window_size = window_size

        # ── TODO: define your layers here ─────────────────────────────────────
        #
        # Example:
        #   self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        #   self.pool  = nn.MaxPool1d(2)
        #   ...
        #   self.fc_out = nn.Linear(<flattened_size>, num_classes)
        #
        raise NotImplementedError(
            "Define the network layers in GestureNet.__init__  "
            "(model/network.py)."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, 1, window_size)``.

        Returns
        -------
        torch.Tensor
            Logit tensor of shape ``(batch, num_classes)``.
        """
        # ── TODO: implement the forward pass ──────────────────────────────────
        #
        # Example:
        #   x = torch.relu(self.conv1(x))
        #   x = self.pool(x)
        #   ...
        #   x = x.flatten(start_dim=1)
        #   return self.fc_out(x)
        #
        raise NotImplementedError(
            "Implement GestureNet.forward in model/network.py."
        )


def build_model(num_classes: int = NUM_CLASSES, window_size: int = WINDOW_SIZE) -> GestureNet:
    """Convenience factory — returns a freshly initialised ``GestureNet``."""
    return GestureNet(num_classes=num_classes, window_size=window_size)
