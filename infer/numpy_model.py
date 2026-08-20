"""
NumPy inference engine.

Loads the weights exported by train/trainer.py and re-implements the
model's forward pass using only numpy — no PyTorch, no ONNX runtime.
This is what runs on the Pi at deployment time.

-----------------------------------------------------------------------
YOUR TASK
-----------------------------------------------------------------------
After you have implemented and trained your GestureCNN (train/model.py),
come back here and implement ``_forward`` so it mirrors your network
exactly.  The weight arrays are named the same as in PyTorch
(e.g. "conv1.weight", "conv1.bias", "fc.weight", "fc.bias").

Provided numpy helpers (already implemented below):
    conv1d(x, weight, bias, padding)  — 1-D convolution
    batch_norm1d(x, gamma, beta, mean, var, eps)
    relu(x)
    max_pool1d(x, kernel_size)
    global_avg_pool(x)
    linear(x, weight, bias)
    softmax(x)

-----------------------------------------------------------------------
"""

import numpy as np
import config


# ---------------------------------------------------------------------------
# numpy layer helpers
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Fully-connected layer: x @ W.T + b"""
    return x @ weight.T + bias


def conv1d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    padding: int = 0,
) -> np.ndarray:
    """
    1-D convolution (stride=1).

    Parameters
    ----------
    x      : (in_channels, length)  — single sample, no batch dim
    weight : (out_channels, in_channels, kernel_size)
    bias   : (out_channels,)
    padding: int — zero-pad each end of the length dimension

    Returns
    -------
    (out_channels, length_out)
    """
    in_ch, length = x.shape
    out_ch, _, ksize = weight.shape

    if padding > 0:
        x = np.pad(x, ((0, 0), (padding, padding)))

    length_out = x.shape[1] - ksize + 1
    out = np.zeros((out_ch, length_out), dtype=np.float32)
    for oc in range(out_ch):
        for ic in range(in_ch):
            for t in range(length_out):
                out[oc, t] += np.dot(weight[oc, ic], x[ic, t:t + ksize])
        out[oc] += bias[oc]
    return out


def max_pool1d(x: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    1-D max pooling (stride = kernel_size, no padding).

    x shape: (channels, length)
    """
    ch, length = x.shape
    length_out = length // kernel_size
    out = np.zeros((ch, length_out), dtype=np.float32)
    for t in range(length_out):
        out[:, t] = x[:, t * kernel_size:(t + 1) * kernel_size].max(axis=1)
    return out


def global_avg_pool(x: np.ndarray) -> np.ndarray:
    """
    Global average pooling over the time dimension.

    x shape: (channels, length)  →  returns (channels,)
    """
    return x.mean(axis=-1)


def batch_norm1d(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Inference-time BatchNorm1d (uses running statistics).

    x shape: (channels, length)
    gamma, beta, running_mean, running_var: (channels,)
    """
    mean = running_mean[:, np.newaxis]
    var = running_var[:, np.newaxis]
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma[:, np.newaxis] * x_norm + beta[:, np.newaxis]


# ---------------------------------------------------------------------------
# Pre-processing (must match train/dataset.py)
# ---------------------------------------------------------------------------

def preprocess(window: list[float]) -> np.ndarray:
    """
    Clamp + min-max normalise a raw distance window.

    Returns shape (1, WINDOW_SAMPLES) — matches the model input.
    """
    arr = np.array(window, dtype=np.float32)
    arr = np.clip(arr, config.MIN_DISTANCE_CM, config.MAX_DISTANCE_CM)
    lo, hi = arr.min(), arr.max()
    span = hi - lo
    if span < 1e-6:
        arr = np.zeros_like(arr)
    else:
        arr = (arr - lo) / span
    return arr.reshape(1, -1)


# ---------------------------------------------------------------------------
# Inference model
# ---------------------------------------------------------------------------

class NumpyGestureModel:
    """
    Runs the gesture CNN forward pass in pure numpy.

    Parameters
    ----------
    weights_path : path to the .npz file produced by train/trainer.py
    """

    def __init__(self, weights_path: str = config.MODEL_WEIGHTS_PATH):
        data = np.load(weights_path)
        self.weights = dict(data)

    def predict(self, window: list[float]) -> tuple[str, float]:
        """
        Classify a single distance window.

        Parameters
        ----------
        window : list of WINDOW_SAMPLES floats (raw cm readings)

        Returns
        -------
        label : the predicted gesture label string
        confidence : softmax probability of the predicted class (0–1)
        """
        x = preprocess(window)              # (1, WINDOW_SAMPLES)
        logits = self._forward(x)           # (NUM_CLASSES,)
        probs = softmax(logits[np.newaxis])[0]
        idx = int(probs.argmax())
        return config.GESTURE_LABELS[idx], float(probs[idx])

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : np.ndarray of shape (1, WINDOW_SAMPLES)

        Returns
        -------
        logits : np.ndarray of shape (NUM_CLASSES,)

        TODO
        ----
        Mirror the architecture you defined in train/model.py using the
        numpy helper functions above.  Retrieve weights with
        ``self.weights["layer.param"]``.

        Example (assumes the suggested skeleton from train/model.py):

            w = self.weights
            x = conv1d(x, w["conv1.weight"], w["conv1.bias"], padding=2)
            x = batch_norm1d(x, w["bn1.weight"], w["bn1.bias"],
                             w["bn1.running_mean"], w["bn1.running_var"])
            x = relu(x)
            x = max_pool1d(x, kernel_size=2)

            x = conv1d(x, w["conv2.weight"], w["conv2.bias"], padding=1)
            x = batch_norm1d(x, w["bn2.weight"], w["bn2.bias"],
                             w["bn2.running_mean"], w["bn2.running_var"])
            x = relu(x)

            x = global_avg_pool(x)          # (32,)
            x = linear(x, w["fc.weight"], w["fc.bias"])  # (NUM_CLASSES,)
            return x
        """
        raise NotImplementedError(
            "Implement _forward in NumpyGestureModel  (infer/numpy_model.py)\n"
            "Mirror the architecture from train/model.py using the numpy helpers."
        )
