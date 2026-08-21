"""
display/ui.py
-------------
Display state machine.

Three states:
    IDLE      → "Ready" / blank  (shown when nothing is happening)
    GESTURE   → the predicted gesture label + a short hold time
    ERROR     → an error message (sensor fault, model not loaded, …)

The state machine is driven by calling ``on_gesture(label)`` or
``on_error(msg)``; it handles the hold time and return-to-idle
transition automatically via ``tick()``.

Usage (inside the main loop)::

    ui = DisplayUI(oled)
    while True:
        prediction = predictor.predict(window)
        if prediction:
            ui.on_gesture(prediction)
        ui.tick()
        time.sleep(0.05)
"""

import time
from enum import Enum, auto
from typing import Optional

from display.oled import OLEDDisplay

# ── Constants ─────────────────────────────────────────────────────────────────

GESTURE_HOLD_S: float = 2.0   # how long to show a gesture label
ERROR_HOLD_S: float = 4.0     # how long to show an error message


# ── State machine ─────────────────────────────────────────────────────────────

class _State(Enum):
    IDLE = auto()
    GESTURE = auto()
    ERROR = auto()


class DisplayUI:
    """
    Manages the OLED display state and transitions.

    Parameters
    ----------
    oled:
        An ``OLEDDisplay`` instance.
    gesture_hold:
        Seconds to display a gesture label before returning to idle.
    error_hold:
        Seconds to display an error before returning to idle.
    """

    def __init__(
        self,
        oled: OLEDDisplay,
        gesture_hold: float = GESTURE_HOLD_S,
        error_hold: float = ERROR_HOLD_S,
    ) -> None:
        self._oled = oled
        self._gesture_hold = gesture_hold
        self._error_hold = error_hold
        self._state = _State.IDLE
        self._expire: float = 0.0
        self._render_idle()

    # ── Public events ─────────────────────────────────────────────────────────

    def on_gesture(self, label: str) -> None:
        """
        Called when the predictor emits a gesture label.

        Transitions to GESTURE state and renders the label.
        """
        self._state = _State.GESTURE
        self._expire = time.monotonic() + self._gesture_hold
        self._render_gesture(label)

    def on_error(self, message: str) -> None:
        """
        Called when a non-fatal error occurs (e.g. sensor read fail).

        Transitions to ERROR state and renders the message.
        """
        self._state = _State.ERROR
        self._expire = time.monotonic() + self._error_hold
        self._render_error(message)

    def tick(self) -> None:
        """
        Must be called on every main-loop iteration.

        Returns to IDLE state once the hold time expires.
        """
        if self._state != _State.IDLE and time.monotonic() >= self._expire:
            self._state = _State.IDLE
            self._render_idle()

    # ── Rendering helpers ─────────────────────────────────────────────────────

    def _render_idle(self) -> None:
        self._oled.show_text(line1="Gesture", line2="Detector", line3="Ready…")

    def _render_gesture(self, label: str) -> None:
        # Show the label in large text; convert underscores to spaces
        friendly = label.replace("_", " ").upper()
        self._oled.show_text(line1=friendly)

    def _render_error(self, message: str) -> None:
        # Truncate to fit
        self._oled.show_text(line1="ERROR", line2=message[:20])
