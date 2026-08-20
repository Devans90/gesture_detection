"""
model.py — PyTorch CNN for 1-D gesture time-series classification.

The network architecture is intentionally left for you to design.
The file provides:
  - a skeleton Module with clear TODO markers
  - the expected input/output contract (documented below)
  - a parameter-count utility so you can compare architectures
  - weight I/O helpers used by train.py and deploy/

Input contract
--------------
  Tensor shape : (batch, 1, WINDOW_SAMPLES)   — single-channel 1-D series
  dtype        : torch.float32
  value range  : whatever your preprocess() function produces

Output contract
---------------
  Tensor shape : (batch, N_CLASSES)  — raw logits (no softmax)
"""

import torch
import torch.nn as nn

from collect import LABELS, WINDOW_SAMPLES

N_CLASSES: int = len(LABELS)


class GestureNet(nn.Module):
    """
    1-D CNN gesture classifier.

    TODO: design and implement the network architecture here.

    Hints / starting points:
      - A small 1-D conv stack works well for short time-series:
            Conv1d → ReLU → MaxPool1d  (×2–3)  → Flatten → Linear → output
      - Keep it tiny — you have ~100–200 training examples and must later
        export weights as plain numpy arrays.
      - Rule of thumb: <10 k parameters is plenty for five gesture classes
        on a 30-sample input.
      - BatchNorm1d after each conv helps stabilise training.
      - Dropout before the final linear layer helps prevent overfitting.

    Suggested minimal skeleton (uncomment and fill in):

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=???, kernel_size=???)
        self.pool  = nn.MaxPool1d(kernel_size=???)
        self.conv2 = nn.Conv1d(???)
        self.fc    = nn.Linear(???, N_CLASSES)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.flatten(1)
            return self.fc(x)
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: define your layers here
        raise NotImplementedError(
            "GestureNet.__init__ is not implemented.\n"
            "Open model.py and design your layer stack."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor, shape (batch, 1, WINDOW_SAMPLES)

        Returns
        -------
        logits : Tensor, shape (batch, N_CLASSES)
        """
        # TODO: implement the forward pass
        raise NotImplementedError(
            "GestureNet.forward is not implemented.\n"
            "Open model.py and write the forward pass."
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_weights(model: nn.Module, path: str) -> None:
    """Save model state dict (used by train.py after training)."""
    torch.save(model.state_dict(), path)


def load_weights(model: nn.Module, path: str, map_location: str = "cpu") -> nn.Module:
    """Load state dict back into a model instance."""
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    net = GestureNet()
    print(f"Parameters: {count_parameters(net):,}")
    dummy = torch.zeros(4, 1, WINDOW_SAMPLES)
    out = net(dummy)
    print(f"Output shape: {out.shape}  (expected [4, {N_CLASSES}])")
