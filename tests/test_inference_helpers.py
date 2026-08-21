"""
tests/test_inference_helpers.py
--------------------------------
Tests for the numpy layer helpers in deploy/inference.py.

These don't require a weights file or a trained model.
"""

import numpy as np
import pytest

from deploy.inference import relu, softmax, max_pool1d, conv1d


def test_relu_clamps_negatives():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    out = relu(x)
    np.testing.assert_array_equal(out, [0.0, 0.0, 1.0, 2.0])


def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, 3.0])
    out = softmax(x)
    assert abs(out.sum() - 1.0) < 1e-6


def test_softmax_numerically_stable():
    x = np.array([1000.0, 1001.0, 1002.0])
    out = softmax(x)
    assert np.isfinite(out).all()


def test_max_pool1d_halves_length():
    x = np.array([[[1, 3, 2, 4, 0, 6]]], dtype=np.float32)  # (1, 1, 6)
    out = max_pool1d(x, kernel_size=2)
    assert out.shape == (1, 1, 3)
    np.testing.assert_array_equal(out[0, 0], [3, 4, 6])


def test_conv1d_output_shape_no_padding():
    batch, in_ch, length = 2, 1, 10
    out_ch, ks = 4, 3
    x = np.random.rand(batch, in_ch, length).astype(np.float32)
    weight = np.random.rand(out_ch, in_ch, ks).astype(np.float32)
    bias = np.zeros(out_ch, dtype=np.float32)
    out = conv1d(x, weight, bias, padding=0)
    assert out.shape == (batch, out_ch, length - ks + 1)


def test_conv1d_output_shape_with_padding():
    batch, in_ch, length = 1, 1, 10
    out_ch, ks, pad = 8, 5, 2
    x = np.random.rand(batch, in_ch, length).astype(np.float32)
    weight = np.random.rand(out_ch, in_ch, ks).astype(np.float32)
    bias = np.zeros(out_ch, dtype=np.float32)
    out = conv1d(x, weight, bias, padding=pad)
    assert out.shape == (batch, out_ch, length)  # same-padding


def test_conv1d_matches_known_result():
    # 1 sample, 1 channel, 5 values; kernel [1, 0, -1] → discrete derivative
    x = np.array([[[1, 2, 4, 7, 11]]], dtype=np.float32)
    weight = np.array([[[1, 0, -1]]], dtype=np.float32)
    bias = np.zeros(1, dtype=np.float32)
    out = conv1d(x, weight, bias, padding=0)
    # [1-4, 2-7, 4-11] = [-3, -5, -7]
    np.testing.assert_array_almost_equal(out[0, 0], [-3, -5, -7])
