"""
Tests for the numpy inference helpers (no model weights needed).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from infer.numpy_model import (
    relu,
    softmax,
    linear,
    conv1d,
    max_pool1d,
    global_avg_pool,
    batch_norm1d,
    preprocess,
)
import config


class TestNumpyHelpers:
    def test_relu_clips_negatives(self):
        x = np.array([-1.0, 0.0, 2.0])
        out = relu(x)
        np.testing.assert_array_equal(out, [0.0, 0.0, 2.0])

    def test_softmax_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0]])
        out = softmax(x)
        assert abs(out.sum() - 1.0) < 1e-6

    def test_linear_shape(self):
        x = np.ones((4,), dtype=np.float32)
        w = np.eye(3, 4, dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)
        out = linear(x, w, b)
        assert out.shape == (3,)

    def test_conv1d_output_shape_no_padding(self):
        x = np.ones((1, 10), dtype=np.float32)
        weight = np.ones((4, 1, 3), dtype=np.float32)
        bias = np.zeros(4, dtype=np.float32)
        out = conv1d(x, weight, bias, padding=0)
        assert out.shape == (4, 8)  # 10 - 3 + 1 = 8

    def test_conv1d_output_shape_with_padding(self):
        x = np.ones((1, 10), dtype=np.float32)
        weight = np.ones((4, 1, 5), dtype=np.float32)
        bias = np.zeros(4, dtype=np.float32)
        out = conv1d(x, weight, bias, padding=2)
        assert out.shape == (4, 10)  # same-padding

    def test_max_pool1d_halves_length(self):
        x = np.arange(16, dtype=np.float32).reshape(2, 8)
        out = max_pool1d(x, kernel_size=2)
        assert out.shape == (2, 4)

    def test_global_avg_pool_shape(self):
        x = np.ones((8, 15), dtype=np.float32) * 3.0
        out = global_avg_pool(x)
        assert out.shape == (8,)
        np.testing.assert_allclose(out, 3.0)


class TestPreprocess:
    def test_output_shape(self):
        window = [20.0] * config.WINDOW_SAMPLES
        out = preprocess(window)
        assert out.shape == (1, config.WINDOW_SAMPLES)

    def test_values_in_zero_one(self):
        import random
        window = [random.uniform(config.MIN_DISTANCE_CM, config.MAX_DISTANCE_CM)
                  for _ in range(config.WINDOW_SAMPLES)]
        out = preprocess(window)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-6

    def test_flat_window_returns_zeros(self):
        window = [15.0] * config.WINDOW_SAMPLES
        out = preprocess(window)
        np.testing.assert_array_equal(out, np.zeros((1, config.WINDOW_SAMPLES)))
