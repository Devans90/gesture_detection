"""
deploy/forward.py — numpy-only inference for ARMv6 Raspberry Pi.

This file re-implements the GestureNet forward pass using only numpy.
No PyTorch, no ONNX runtime, no ML framework required on the Pi.

YOU MUST keep this file in sync with model.py — if you add or change
layers in GestureNet, update the corresponding numpy ops here.

Everything that is architecture-specific is marked  ← TODO.

Usage (on Pi):
    from deploy.forward import load_model, predict

    model = load_model("deploy/weights.npz")
    label, confidence = predict(model, distance_array)
"""

from pathlib import Path
from typing import Dict, NamedTuple

import numpy as np

# These are copied literally from collect.py / dataset.py to avoid imports
from collect import LABELS, WINDOW_SAMPLES

CONFIDENCE_THRESHOLD: float = 0.7  # predictions below this are reported as "idle"


# ---------------------------------------------------------------------------
# Numpy primitives  (add more as your architecture needs them)
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def conv1d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int = 1) -> np.ndarray:
    """
    1-D convolution — pure numpy, no padding.

    Parameters
    ----------
    x      : (C_in, L)
    weight : (C_out, C_in, K)
    bias   : (C_out,)

    Returns
    -------
    out : (C_out, L_out)  where L_out = (L - K) // stride + 1
    """
    c_out, c_in, k = weight.shape
    l_in = x.shape[1]
    l_out = (l_in - k) // stride + 1
    out = np.zeros((c_out, l_out), dtype=np.float32)
    for o in range(c_out):
        for i in range(c_in):
            for j in range(l_out):
                out[o, j] += np.sum(x[i, j * stride: j * stride + k] * weight[o, i])
        out[o] += bias[o]
    return out


def maxpool1d(x: np.ndarray, kernel_size: int, stride: int = None) -> np.ndarray:
    """
    1-D max-pooling.

    Parameters
    ----------
    x           : (C, L)
    kernel_size : int
    stride      : int, defaults to kernel_size (non-overlapping)

    Returns
    -------
    out : (C, L_out)
    """
    if stride is None:
        stride = kernel_size
    c, l_in = x.shape
    l_out = (l_in - kernel_size) // stride + 1
    out = np.zeros((c, l_out), dtype=np.float32)
    for j in range(l_out):
        out[:, j] = x[:, j * stride: j * stride + kernel_size].max(axis=1)
    return out


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Fully-connected layer.

    Parameters
    ----------
    x      : (in_features,)
    weight : (out_features, in_features)
    bias   : (out_features,)
    """
    return weight @ x + bias


def batchnorm1d_eval(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    BatchNorm1d in inference mode (uses running statistics, not batch stats).

    x shape: (C, L)  — applies per-channel normalisation along L.
    """
    x_norm = (x - running_mean[:, None]) / np.sqrt(running_var[:, None] + eps)
    return weight[:, None] * x_norm + bias[:, None]


# ---------------------------------------------------------------------------
# Model weights container
# ---------------------------------------------------------------------------

class GestureNetWeights(NamedTuple):
    """
    Holds all numpy weight arrays extracted from the .npz file.

    TODO: add a field here for every nn.Parameter in your GestureNet.
    The field names should match the PyTorch state_dict keys exactly,
    with '.' replaced by '_' (numpy .npz stores them as strings anyway
    so you can access them however you like — this NamedTuple just makes
    the forward function type-safe and readable).

    Example fields if you used conv1, pool, conv2, fc:
        conv1_weight : np.ndarray
        conv1_bias   : np.ndarray
        conv2_weight : np.ndarray
        conv2_bias   : np.ndarray
        fc_weight    : np.ndarray
        fc_bias      : np.ndarray
    """
    # TODO: declare your weight fields here
    pass  # remove this line once you've added your fields


def load_model(weights_path: str) -> GestureNetWeights:
    """
    Load exported .npz weights into a GestureNetWeights tuple.

    TODO: update to match your GestureNetWeights fields.
    """
    npz = np.load(weights_path)

    # TODO: replace with your actual field names, e.g.:
    # return GestureNetWeights(
    #     conv1_weight=npz["conv1.weight"],
    #     conv1_bias=npz["conv1.bias"],
    #     ...
    # )
    raise NotImplementedError(
        "load_model() is not implemented.\n"
        "Open deploy/forward.py, declare your GestureNetWeights fields, "
        "and fill in load_model() to match your architecture."
    )


# ---------------------------------------------------------------------------
# Forward pass  ← YOU IMPLEMENT THIS
# ---------------------------------------------------------------------------

def _forward(w: GestureNetWeights, x: np.ndarray) -> np.ndarray:
    """
    Run the numpy forward pass.

    Parameters
    ----------
    w : GestureNetWeights — weight arrays loaded from .npz
    x : np.ndarray, shape (1, WINDOW_SAMPLES) — preprocessed input

    Returns
    -------
    logits : np.ndarray, shape (N_CLASSES,) — raw scores, no softmax applied

    TODO: implement this to match your GestureNet forward() method,
    using the numpy primitives defined above (conv1d, relu, maxpool1d, linear …).

    Example skeleton (uncomment and adapt):

        x = relu(conv1d(x, w.conv1_weight, w.conv1_bias))
        x = maxpool1d(x, kernel_size=2)
        x = relu(conv1d(x, w.conv2_weight, w.conv2_bias))
        x = maxpool1d(x, kernel_size=2)
        x = x.flatten()
        return linear(x, w.fc_weight, w.fc_bias)
    """
    raise NotImplementedError(
        "_forward() is not implemented.\n"
        "Open deploy/forward.py and translate your model.py forward() "
        "into numpy operations."
    )


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

def predict(w: GestureNetWeights, distance_window: np.ndarray) -> tuple:
    """
    Classify a single gesture window.

    Parameters
    ----------
    w               : loaded model weights
    distance_window : 1-D float32 array of length WINDOW_SAMPLES,
                      already preprocessed (same transform as dataset.preprocess)

    Returns
    -------
    label      : str  — predicted gesture name, or "idle" if below threshold
    confidence : float — max softmax probability [0, 1]
    """
    x = distance_window.reshape(1, WINDOW_SAMPLES).astype(np.float32)
    logits = _forward(w, x)
    probs = softmax(logits)
    idx = int(probs.argmax())
    confidence = float(probs[idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return "idle", confidence
    return LABELS[idx], confidence
