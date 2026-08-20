"""
main.py
-------
Boot entry point — wires sensor, buffer, predictor, and display together.

Intended to be launched by the systemd unit on boot.

Architecture
~~~~~~~~~~~~
::

    HC_SR04
      │  raw distance (cm or None)
      ▼
    RollingBuffer
      │  forwards to GestureSegmenter on every push
      ▼
    GestureSegmenter
      │  fires on_window(window) when motion detected
      ▼
    Predictor
      │  returns gesture label or None
      ▼
    DisplayUI
         renders label on OLED or stays idle

Prerequisites
~~~~~~~~~~~~~
Before this will work end-to-end you need to:
1.  Implement ``_motion_detected`` in ``capture/buffer.py``.
2.  Implement ``GestureNet`` in ``model/network.py``.
3.  Train the model and export weights to ``deploy/weights.npz``.
4.  Implement ``NumpyInference.predict`` in ``deploy/inference.py``.
"""

import signal
import sys
import time

from capture.sensor import HC_SR04, SAMPLE_RATE_HZ
from capture.buffer import RollingBuffer, GestureSegmenter, WINDOW_SIZE
from deploy.predictor import Predictor
from display.oled import OLEDDisplay
from display.ui import DisplayUI

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

WEIGHTS_PATH: str = "deploy/weights.npz"
SAMPLE_INTERVAL: float = 1.0 / SAMPLE_RATE_HZ


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # ── Hardware init ─────────────────────────────────────────────────────────
    oled = OLEDDisplay()
    ui = DisplayUI(oled)
    sensor = HC_SR04()
    sensor.setup()

    # ── Model init ────────────────────────────────────────────────────────────
    try:
        predictor = Predictor(weights_path=WEIGHTS_PATH)
    except (FileNotFoundError, NotImplementedError) as exc:
        ui.on_error(str(exc)[:40])
        predictor = None  # type: ignore[assignment]

    # ── Buffer + segmenter ────────────────────────────────────────────────────
    buffer = RollingBuffer(maxlen=WINDOW_SIZE)

    def _on_window(window: np.ndarray) -> None:
        """Called by GestureSegmenter when a gesture window is ready."""
        if predictor is None:
            return
        try:
            label = predictor.predict(window)
        except NotImplementedError:
            ui.on_error("predict not impl")
            return

        if label is not None:
            print(f"[main] gesture: {label}")
            ui.on_gesture(label)

    segmenter = GestureSegmenter(buffer=buffer, on_gesture=_on_window)

    # ── Shutdown handler ──────────────────────────────────────────────────────
    def _shutdown(signum, frame):  # noqa: ANN001
        print("[main] Shutting down…")
        sensor.teardown()
        oled.clear()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Main loop ─────────────────────────────────────────────────────────────
    print("[main] Running.  Press Ctrl-C to exit.")
    while True:
        distance = sensor.read()
        buffer.push(distance)

        try:
            segmenter.update()
        except NotImplementedError:
            ui.on_error("buffer not impl")
            time.sleep(1)
            continue

        ui.tick()
        time.sleep(SAMPLE_INTERVAL)


if __name__ == "__main__":
    main()
