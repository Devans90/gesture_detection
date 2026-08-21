"""
tests/test_buffer.py
--------------------
Unit tests for RollingBuffer.

GestureSegmenter tests are omitted here because _motion_detected raises
NotImplementedError until the user implements it.
"""

import numpy as np
import pytest

from capture.buffer import RollingBuffer, WINDOW_SIZE


def test_buffer_fill_and_full():
    buf = RollingBuffer(maxlen=4)
    assert not buf.full()
    for v in [10.0, 20.0, 30.0, 40.0]:
        buf.push(v)
    assert buf.full()


def test_buffer_none_forward_fill():
    buf = RollingBuffer(maxlen=4, idle_fill=999.0)
    buf.push(100.0)
    buf.push(None)  # should forward-fill with 100.0
    arr = buf.as_array()
    assert arr[0] == pytest.approx(100.0)
    assert arr[1] == pytest.approx(100.0)


def test_buffer_idle_fill_before_any_valid():
    buf = RollingBuffer(maxlen=3, idle_fill=400.0)
    buf.push(None)
    buf.push(None)
    arr = buf.as_array()
    assert (arr == 400.0).all()


def test_buffer_as_array_dtype():
    buf = RollingBuffer(maxlen=3)
    buf.push(50.0)
    buf.push(60.0)
    buf.push(70.0)
    arr = buf.as_array()
    assert arr.dtype == np.float32


def test_buffer_reset():
    buf = RollingBuffer(maxlen=4)
    for v in [1.0, 2.0, 3.0, 4.0]:
        buf.push(v)
    buf.reset()
    assert len(buf) == 0
    assert not buf.full()
