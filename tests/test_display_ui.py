"""
tests/test_display_ui.py
------------------------
Unit tests for the display state machine (no real OLED needed).
"""

import time
import pytest

from display.oled import OLEDDisplay
from display.ui import DisplayUI, _State


class _FakeOLED(OLEDDisplay):
    """Stub that records calls instead of driving hardware."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.width = 128
        self.height = 64
        self._device = None

    def show_text(self, line1="", line2="", line3="", font_size=14):
        self.calls.append(("show", line1, line2, line3))

    def clear(self):
        self.calls.append(("clear",))


def _make_ui(gesture_hold=10.0, error_hold=10.0):
    oled = _FakeOLED()
    ui = DisplayUI(oled, gesture_hold=gesture_hold, error_hold=error_hold)
    return ui, oled


def test_initial_state_is_idle():
    ui, _ = _make_ui()
    assert ui._state == _State.IDLE


def test_on_gesture_transitions_to_gesture():
    ui, oled = _make_ui()
    ui.on_gesture("swipe_left")
    assert ui._state == _State.GESTURE
    # Check the label was rendered
    assert any("SWIPE LEFT" in str(call) for call in oled.calls)


def test_on_error_transitions_to_error():
    ui, _ = _make_ui()
    ui.on_error("sensor fault")
    assert ui._state == _State.ERROR


def test_tick_returns_to_idle_after_hold():
    ui, _ = _make_ui(gesture_hold=0.01)
    ui.on_gesture("wave")
    time.sleep(0.05)
    ui.tick()
    assert ui._state == _State.IDLE


def test_tick_does_not_return_early():
    ui, _ = _make_ui(gesture_hold=10.0)
    ui.on_gesture("push")
    ui.tick()  # called immediately — should not return to idle yet
    assert ui._state == _State.GESTURE
