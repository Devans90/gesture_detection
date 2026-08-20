"""
Gesture CNN model definition.

The network takes a single-channel time-series of shape (1, WINDOW_SAMPLES)
and outputs class logits of shape (NUM_CLASSES,).

Architecture overview (skeleton — forward pass is YOUR task to implement)
--------------------------------------------------------------------------
  Conv1d block 1  →  Conv1d block 2  →  global-avg-pool  →  FC  →  logits

Each Conv1d block should include:
    Conv1d → BatchNorm1d → ReLU → (optional) MaxPool1d

Keep the model small!  The Pi is ARMv6 with no ML runtime; you will be
running a numpy re-implementation of this exact architecture at inference
time.  Every layer you add here means more numpy to write later.

-----------------------------------------------------------------------
YOUR TASK
-----------------------------------------------------------------------
1.  In `__init__`, define your layers (self.conv1, self.bn1, …).
2.  In `forward`, pass `x` through those layers and return logits.
3.  Keep the architecture consistent with infer/numpy_model.py so you
    can copy the weights accurately.
-----------------------------------------------------------------------
"""

import config

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class GestureCNN(nn.Module):
        """
        Lightweight 1-D CNN for gesture classification.

        Input  shape: (batch, 1, WINDOW_SAMPLES)
        Output shape: (batch, NUM_CLASSES)
        """

        def __init__(
            self,
            num_classes: int = config.NUM_CLASSES,
            window_samples: int = config.WINDOW_SAMPLES,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.window_samples = window_samples

            # ----------------------------------------------------------
            # TODO: define your layers here.
            #
            # Suggested skeleton (adjust channels/kernel sizes freely):
            #
            #   self.conv1  = nn.Conv1d(in_channels=1,  out_channels=16,
            #                           kernel_size=5, padding=2)
            #   self.bn1    = nn.BatchNorm1d(16)
            #   self.pool1  = nn.MaxPool1d(kernel_size=2)
            #
            #   self.conv2  = nn.Conv1d(in_channels=16, out_channels=32,
            #                           kernel_size=3, padding=1)
            #   self.bn2    = nn.BatchNorm1d(32)
            #   self.pool2  = nn.MaxPool1d(kernel_size=2)
            #
            #   self.fc     = nn.Linear(32, num_classes)
            #
            # After two MaxPool1d(2) on a WINDOW_SAMPLES=30 series:
            #   30 → 15 → 7  (floor division)
            # So the FC input dim = 32 * 7 = 224 if you use global max/avg
            # pool instead, input dim = 32.
            # ----------------------------------------------------------
            raise NotImplementedError(
                "Define your layers in GestureCNN.__init__  (train/model.py)"
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Parameters
            ----------
            x : Tensor of shape (batch, 1, WINDOW_SAMPLES)

            Returns
            -------
            logits : Tensor of shape (batch, NUM_CLASSES)

            TODO
            ----
            Pass `x` through the layers you defined in __init__.
            Example pattern:

                x = F.relu(self.bn1(self.conv1(x)))
                x = self.pool1(x)
                x = F.relu(self.bn2(self.conv2(x)))
                x = x.mean(dim=-1)   # global average pool
                x = self.fc(x)
                return x
            """
            raise NotImplementedError(
                "Implement the forward pass in GestureCNN.forward  (train/model.py)"
            )

        # ------------------------------------------------------------------
        # Utility
        # ------------------------------------------------------------------

        def count_parameters(self) -> int:
            """Return the total number of trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:
    # Provide a placeholder so the rest of the codebase can import this
    # module even without PyTorch (e.g. on the inference Pi).
    class GestureCNN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate GestureCNN.")
