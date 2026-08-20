"""
main.py — boot-on-power gesture detection loop.

This is the entry-point that runs automatically on Pi startup via systemd.
It ties together:
  - capture.py  — HC-SR04 distance stream
  - dataset.py  — preprocess() for the incoming window
  - deploy/forward.py — numpy-only inference
  - display.py  — SSD1306 OLED output

State machine
-------------
IDLE      → watching the sensor; hand not detected
RECORDING → hand entered trigger zone; accumulating WINDOW_SAMPLES
CLASSIFYING → window full; run forward pass, display result
COOLDOWN  → brief pause after a prediction to avoid rapid re-triggering

         hand enters zone              window full
  IDLE ────────────────► RECORDING ──────────────► CLASSIFYING
   ▲                                                    │
   └────────────────────────────────────────────────────┘
              after COOLDOWN_S seconds

Usage:
    python main.py
    python main.py --weights deploy/weights.npz --threshold 0.7
"""

import argparse
import time
import traceback
from collections import deque
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional

import numpy as np

import capture
import dataset
from deploy.forward import load_model, predict, CONFIDENCE_THRESHOLD
from display import Display

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: str = "deploy/weights.npz"
TRIGGER_CM: float = 50.0       # hand must be closer than this to start recording
COOLDOWN_S: float = 2.0        # seconds to wait after a prediction before re-arming

WINDOW_SAMPLES: int = capture.WINDOW_SAMPLES  # from collect via capture


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE = auto()
    RECORDING = auto()
    CLASSIFYING = auto()
    COOLDOWN = auto()


def run_loop(weights_path: str, threshold: float) -> None:
    """Main inference loop — runs until KeyboardInterrupt or fatal error."""

    try:
        import RPi.GPIO as GPIO  # type: ignore
    except ImportError:
        print("RPi.GPIO not available — running in mock mode (no real sensor data)")
        import types
        GPIO = types.SimpleNamespace(
            BCM=0, OUT=0, IN=0,
            setmode=lambda _: None,
            setup=lambda *_: None,
            output=lambda *_: None,
            input=lambda _: 0,
            cleanup=lambda: None,
        )

    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Run deploy/export.py after training to create this file."
        )

    weights = load_model(weights_path)
    capture.setup_gpio(GPIO)

    with Display() as disp:
        disp.show_message("Gesture", "Detection v1")
        time.sleep(1.0)
        disp.show_idle()

        state = State.IDLE
        window: List[Optional[float]] = []
        cooldown_until: float = 0.0

        print("Ready.  (Ctrl-C to stop)\n")

        try:
            for dist in capture.sample_stream(GPIO):
                now = time.monotonic()

                # ---- COOLDOWN ----------------------------------------
                if state == State.COOLDOWN:
                    if now >= cooldown_until:
                        state = State.IDLE
                        disp.show_idle()
                    continue

                # ---- IDLE → RECORDING --------------------------------
                if state == State.IDLE:
                    if dist is not None and dist < TRIGGER_CM:
                        state = State.RECORDING
                        window = [dist]
                    continue

                # ---- RECORDING ---------------------------------------
                if state == State.RECORDING:
                    window.append(dist)
                    if len(window) >= WINDOW_SAMPLES:
                        state = State.CLASSIFYING

                # ---- CLASSIFYING -------------------------------------
                if state == State.CLASSIFYING:
                    # Fill gaps and preprocess
                    from collect import _interpolate_gaps
                    cleaned = _interpolate_gaps(window[:WINDOW_SAMPLES])
                    arr = np.array(cleaned, dtype=np.float32)

                    try:
                        arr = dataset.preprocess(arr.reshape(1, -1)).flatten()
                        label, confidence = predict(weights, arr)
                        print(f"  → {label}  ({confidence:.0%})")
                        disp.show_label(label, confidence)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  Inference error: {exc}")
                        disp.show_message("Error", str(exc)[:20])

                    cooldown_until = time.monotonic() + COOLDOWN_S
                    state = State.COOLDOWN
                    window = []

        except KeyboardInterrupt:
            print("\nStopped.")

    GPIO.cleanup()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Gesture detection loop")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help="Confidence threshold below which prediction is 'idle'")
    args = parser.parse_args()
    run_loop(args.weights, args.threshold)


if __name__ == "__main__":
    main()
