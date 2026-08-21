"""
deploy/inference.py
-------------------
Numpy-only forward pass — no PyTorch, no ONNX runtime.

ARMv6 (Raspberry Pi 1 B+) has no pip-installable PyTorch or ONNX runtime, so
the trained model's weights are exported from PyTorch and re-implemented here
using only numpy.

Workflow
~~~~~~~~
1. **On your laptop**: train the model and export weights::

       python -m deploy.export --checkpoint runs/<id>/best.pt \\
                               --output deploy/weights.npz

2. **Copy to Pi**: ``scp deploy/weights.npz pi@raspberrypi.local:gesture/deploy/``

3. **On the Pi**: ``NumpyInference`` loads ``weights.npz`` and runs the forward
   pass using only numpy.

This file contains:
* ``NumpyInference`` — weight loader + ``predict`` stub (**implement forward pass**).
* ``export_weights`` — helper to extract weights from a PyTorch checkpoint.

Your task
~~~~~~~~~
Once ``GestureNet.forward`` is implemented, translate each layer into numpy
operations inside ``NumpyInference.predict``.  The docstring explains the
expected shapes at each step.
"""

from pathlib import Path
from typing import Union

import numpy as np


# ── Weight export (run on laptop with PyTorch) ────────────────────────────────

def export_weights(checkpoint_path: str, output_path: str = "deploy/weights.npz") -> None:
    """
    Extract all parameters from a PyTorch ``.pt`` checkpoint and save them
    as a ``.npz`` file that can be loaded on the Pi without PyTorch.

    Parameters
    ----------
    checkpoint_path:
        Path to ``best.pt`` produced by ``model/train.py``.
    output_path:
        Where to write the ``.npz`` file.

    Run this **on your laptop** (requires PyTorch) and then copy the output
    to the Pi.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("export_weights requires PyTorch — run on your laptop.") from e

    state = torch.load(checkpoint_path, map_location="cpu")
    arrays = {k: v.numpy() for k, v in state.items()}
    np.savez(output_path, **arrays)
    print(f"[export] Saved {len(arrays)} weight arrays → {output_path}")


# ── Numpy inference ───────────────────────────────────────────────────────────

class NumpyInference:
    """
    Runs the trained gesture CNN using only numpy.

    Parameters
    ----------
    weights_path:
        Path to the ``.npz`` file produced by ``export_weights``.
    """

    def __init__(self, weights_path: Union[str, Path] = "deploy/weights.npz") -> None:
        self._weights_path = Path(weights_path)
        self._weights: dict[str, np.ndarray] = {}
        self._load_weights()

    def _load_weights(self) -> None:
        """Load the .npz file into a name → array dictionary."""
        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {self._weights_path}\n"
                "Run deploy/inference.py::export_weights on your laptop first."
            )
        npz = np.load(self._weights_path)
        self._weights = {k: npz[k] for k in npz.files}
        print(f"[NumpyInference] Loaded {len(self._weights)} weight arrays from {self._weights_path}")

    def predict(self, window: np.ndarray) -> np.ndarray:
        """
        Run the forward pass and return class logits.

        Parameters
        ----------
        window:
            1-D float32 array of shape ``(WINDOW_SIZE,)``, already normalised
            to [0, 1] (use ``NormaliseWindow`` from ``model/dataset.py``).

        Returns
        -------
        np.ndarray
            1-D array of shape ``(NUM_CLASSES,)`` — raw logits (not softmax).
            Call ``np.argmax`` to get the predicted class index.

        Implementation guide
        ~~~~~~~~~~~~~~~~~~~~
        Translate each layer of ``GestureNet`` into numpy.  Helper functions
        for common operations are below; use them or write your own.

        Typical skeleton (adjust to match your architecture)::

            # Reshape to (1, 1, WINDOW_SIZE) matching Conv1d input
            x = window.reshape(1, 1, -1)

            # Conv block 1
            x = conv1d(x, self._weights["conv1.weight"], self._weights["conv1.bias"])
            x = relu(x)

            # ... more layers ...

            # Final linear layer → logits
            x = x.flatten()
            logits = x @ self._weights["fc_out.weight"].T + self._weights["fc_out.bias"]
            return logits
        """
        # ── TODO: implement the numpy forward pass ────────────────────────────
        raise NotImplementedError(
            "Implement NumpyInference.predict in deploy/inference.py.\n"
            "Translate each layer of GestureNet into numpy operations.\n"
            "See the docstring above for a step-by-step guide."
        )


# ── Numpy layer helpers ───────────────────────────────────────────────────────
# These are provided for you — use them inside predict().

def relu(x: np.ndarray) -> np.ndarray:
    """Element-wise ReLU."""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def max_pool1d(x: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    1-D max pooling along the last axis.

    Parameters
    ----------
    x:
        Shape ``(batch, channels, length)``.
    kernel_size:
        Pooling window; output length = length // kernel_size.
    """
    batch, ch, length = x.shape
    out_len = length // kernel_size
    x_trimmed = x[:, :, :out_len * kernel_size]
    return x_trimmed.reshape(batch, ch, out_len, kernel_size).max(axis=-1)


def conv1d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    padding: int = 0,
) -> np.ndarray:
    """
    Naive 1-D convolution (no stride, dilation=1).

    Parameters
    ----------
    x:
        Input of shape ``(batch, in_channels, length)``.
    weight:
        Conv weight of shape ``(out_channels, in_channels, kernel_size)``.
    bias:
        Bias of shape ``(out_channels,)``.
    padding:
        Zero-padding on both sides of the length axis.

    Returns
    -------
    np.ndarray
        Shape ``(batch, out_channels, length + 2*padding - kernel_size + 1)``.

    Note
    ----
    This is intentionally simple and not optimised.  For a 40-sample window
    with small filters the speed is fine on Pi; if you hit latency issues,
    use ``scipy.signal.correlate`` or write a Cython extension.
    """
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding)))

    batch, in_ch, length = x.shape
    out_ch, in_ch_w, ks = weight.shape
    out_len = length - ks + 1
    out = np.zeros((batch, out_ch, out_len), dtype=np.float32)

    for k in range(ks):
        # x[:, :, k:k+out_len] has shape (batch, in_ch, out_len)
        # weight[:, :, k]       has shape (out_ch, in_ch)
        out += np.einsum("bci,oi->boi", x[:, :, k:k + out_len], weight[:, :, k])

    out += bias[np.newaxis, :, np.newaxis]
    return out
