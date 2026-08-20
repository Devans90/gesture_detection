"""
tests/test_sensor.py
--------------------
Unit tests for the HC-SR04 sensor module.

These run without GPIO hardware (the sensor stub returns None on non-Pi).
"""

import pytest
from capture.sensor import HC_SR04, MIN_DISTANCE_CM, MAX_DISTANCE_CM


def test_sensor_instantiates():
    s = HC_SR04()
    assert s.trig_pin == 23
    assert s.echo_pin == 24


def test_sensor_read_returns_none_without_gpio():
    """On a non-Pi host, read() must return None (no GPIO available)."""
    s = HC_SR04()
    s.setup()
    result = s.read()
    # On Pi this would be a float; on CI it must be None
    assert result is None or (MIN_DISTANCE_CM <= result <= MAX_DISTANCE_CM)
    s.teardown()


def test_sensor_custom_pins():
    s = HC_SR04(trig_pin=17, echo_pin=27)
    assert s.trig_pin == 17
    assert s.echo_pin == 27
