"""
Labelled gesture data collector.

Run this script on the Pi to record training examples.  Each session
produces a separate CSV file under data/raw/ so you can hold out whole
sessions during validation.

Usage
-----
    python -m collect.recorder --gesture swipe_left --session 1 --reps 20

The script will:
  1. Show a countdown before each repetition.
  2. Capture exactly WINDOW_SAMPLES of distance data.
  3. Append the labelled row to the session CSV.
  4. Pause briefly so you can reset your hand position.
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from hardware.sensor import HCSR04


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _countdown(seconds: int = 3) -> None:
    """Print a countdown so the user can prepare."""
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end=" ", flush=True)
        time.sleep(1)
    print("GO!", flush=True)


def _capture_window(sensor: HCSR04) -> list[float]:
    """
    Capture exactly WINDOW_SAMPLES readings at SAMPLE_RATE_HZ.

    Dropped readings are forward-filled from the last valid sample.
    """
    period = 1.0 / config.SAMPLE_RATE_HZ
    samples: list[float] = []
    last_valid = config.MAX_DISTANCE_CM

    while len(samples) < config.WINDOW_SAMPLES:
        t0 = time.monotonic()
        raw = sensor.read_cm()
        if raw is None:
            samples.append(last_valid)
        else:
            samples.append(raw)
            last_valid = raw
        elapsed = time.monotonic() - t0
        sleep_for = period - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    return samples


def _output_path(gesture: str, session: int) -> str:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    return os.path.join(config.DATA_DIR, f"{gesture}_session{session:02d}.csv")


def _write_row(path: str, label: str, samples: list[float], rep: int) -> None:
    """Append one labelled row to the CSV (create with header if new)."""
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            # Header: label, rep, d_0, d_1, ..., d_(N-1)
            header = ["label", "rep"] + [f"d_{i}" for i in range(config.WINDOW_SAMPLES)]
            writer.writerow(header)
        writer.writerow([label, rep] + samples)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Record labelled gesture examples.")
    parser.add_argument(
        "--gesture",
        required=True,
        choices=config.GESTURE_LABELS,
        help="Gesture label to record.",
    )
    parser.add_argument(
        "--session",
        type=int,
        default=1,
        help="Session ID (use a different number for each recording sitting).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=20,
        help="Number of repetitions to record.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use simulated sensor readings (for off-Pi development).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.5,
        help="Seconds to pause between repetitions.",
    )
    args = parser.parse_args()

    sensor = HCSR04(simulate=args.simulate)
    out_path = _output_path(args.gesture, args.session)

    print(f"\nRecording gesture: '{args.gesture}'  session={args.session}")
    print(f"Output: {out_path}")
    print(f"Reps: {args.reps}  |  Window: {config.WINDOW_SAMPLES} samples "
          f"@ {config.SAMPLE_RATE_HZ} Hz  ({config.WINDOW_SECONDS:.1f}s)\n")

    try:
        for rep in range(1, args.reps + 1):
            print(f"[{rep}/{args.reps}] Get ready...", end=" ")
            _countdown(3)

            samples = _capture_window(sensor)
            _write_row(out_path, args.gesture, samples, rep)

            print(f"  Captured {len(samples)} samples. "
                  f"Range: {min(samples):.1f}–{max(samples):.1f} cm")

            if rep < args.reps:
                time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\nRecording interrupted.")
    finally:
        sensor.cleanup()

    print(f"\nDone. Saved to {out_path}")


if __name__ == "__main__":
    main()
