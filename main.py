"""
Main runtime loop.

Runs on the Pi on boot, continuously samples the HC-SR04, classifies
gestures and updates the OLED display.

Run directly:
    python main.py

Or via the systemd service (see deploy/gesture.service).
"""

import signal
import sys
import time

import config
from capture.sampler import Sampler
from display.oled import OLEDDisplay
from hardware.sensor import HCSR04
from infer.numpy_model import NumpyGestureModel


# ---------------------------------------------------------------------------
# Idle detection
# ---------------------------------------------------------------------------

def _is_idle(window: list[float]) -> bool:
    """
    Return True if no hand is likely present in this window.

    The heuristic: if all samples are above IDLE_THRESHOLD_CM the window
    is considered idle — nothing interesting happened.

    You may want to replace or extend this once you have real data; e.g.
    check the standard deviation of the window for a more robust test.
    """
    return all(v >= config.IDLE_THRESHOLD_CM for v in window)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class Runtime:
    """Ties the sampler, model, and display together."""

    def __init__(self, simulate_sensor: bool = False, simulate_display: bool = False):
        self.display = OLEDDisplay(simulate=simulate_display)
        self.display.show_message("Loading model…")

        self.model = NumpyGestureModel(config.MODEL_WEIGHTS_PATH)

        sensor = HCSR04(simulate=simulate_sensor)
        self.sampler = Sampler(
            sensor=sensor,
            on_window=self._on_window,
            overlap=0.5,   # 50 % overlap for faster response
        )

        self.display.show_idle()
        self._last_label: str | None = None

    def run(self) -> None:
        """Start sampling.  Blocks until SIGINT / SIGTERM."""
        def _shutdown(sig, frame):
            print("\nShutting down…")
            self.sampler.stop()
            self.display.show_message("Bye!")
            time.sleep(1)
            self.display.clear()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        print("Gesture detector running.  Press Ctrl-C to stop.")
        self.sampler.run()

    # ------------------------------------------------------------------
    # Window callback
    # ------------------------------------------------------------------

    def _on_window(self, window: list[float]) -> None:
        """Called by Sampler every time a new window is ready."""
        if _is_idle(window):
            if self._last_label is not None:
                self.display.show_idle()
                self._last_label = None
            return

        label, confidence = self.model.predict(window)

        if label != self._last_label:
            self.display.show_gesture(label, confidence)
            self._last_label = label
            print(f"Gesture: {label}  ({confidence * 100:.0f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gesture detection runtime.")
    parser.add_argument("--simulate", action="store_true",
                        help="Simulate both sensor and display (off-Pi development).")
    args = parser.parse_args()

    runtime = Runtime(simulate_sensor=args.simulate, simulate_display=args.simulate)
    runtime.run()
