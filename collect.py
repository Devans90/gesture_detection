"""
collect.py — interactive labelled-data collection CLI.

Run on the Pi to record gesture examples.  Each session creates a timestamped
CSV file under data/raw/.

Session file format (CSV):
    timestamp_ms, d0, d1, ..., d{WINDOW_SAMPLES-1}, label

Usage:
    python collect.py --label swipe_left --sessions 10
    python collect.py --list-labels
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import List, Optional

import capture  # local module

# ---------------------------------------------------------------------------
# Collection parameters
# ---------------------------------------------------------------------------
WINDOW_SECONDS: float = 1.5     # length of one gesture window
WINDOW_SAMPLES: int = int(capture.SAMPLE_RATE_HZ * WINDOW_SECONDS)  # ~30

# Gesture is triggered when hand enters the detection zone
TRIGGER_CM: float = 50.0        # distance below which recording starts
PRE_ROLL_SAMPLES: int = 3       # samples captured before threshold crossing

DATA_DIR: Path = Path("data/raw")

LABELS: List[str] = [
    "swipe_left",
    "swipe_right",
    "push_down",
    "pull_up",
    "idle",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(label: str) -> Path:
    ts = int(time.time())
    return DATA_DIR / f"{label}_{ts}.csv"


def _interpolate_gaps(window: List[Optional[float]]) -> List[float]:
    """
    Replace None values (dropped readings) with linear interpolation.

    Edges that are still None after neighbour search are filled with the
    nearest valid value (forward/backward fill).
    """
    result: List[Optional[float]] = list(window)
    n = len(result)

    # Forward fill leading Nones
    first_valid = next((i for i, v in enumerate(result) if v is not None), None)
    if first_valid is None:
        return [0.0] * n  # entire window is empty — discard upstream
    for i in range(first_valid):
        result[i] = result[first_valid]

    # Backward fill trailing Nones
    last_valid = next((i for i, v in enumerate(reversed(result)) if v is not None), None)
    last_valid_idx = n - 1 - last_valid  # type: ignore[operator]
    for i in range(last_valid_idx + 1, n):
        result[i] = result[last_valid_idx]

    # Linear interpolation for interior Nones
    i = 0
    while i < n:
        if result[i] is None:
            j = i
            while j < n and result[j] is None:
                j += 1
            # result[i-1] is valid (we filled leading Nones above)
            v_left: float = result[i - 1]  # type: ignore[assignment]
            v_right: float = result[j] if j < n else result[i - 1]  # type: ignore
            for k in range(i, j):
                frac = (k - i + 1) / (j - i + 1)
                result[k] = v_left + frac * (v_right - v_left)
            i = j
        else:
            i += 1

    return result  # type: ignore[return-value]


def collect_one(GPIO, label: str) -> Optional[List[float]]:
    """
    Wait for a gesture trigger, record WINDOW_SAMPLES readings, return window.

    Returns None if the window is entirely empty (sensor failure).
    """
    print(f"  Waiting for gesture (hand closer than {TRIGGER_CM} cm)…", end="", flush=True)

    ring: List[Optional[float]] = [None] * PRE_ROLL_SAMPLES
    ring_idx = 0

    for dist in capture.sample_stream(GPIO):
        ring[ring_idx % PRE_ROLL_SAMPLES] = dist
        ring_idx += 1

        if dist is not None and dist < TRIGGER_CM:
            break

    print(" triggered!")

    # Collect the main window, seeded with the pre-roll
    window: List[Optional[float]] = list(ring[-PRE_ROLL_SAMPLES:]) if ring_idx >= PRE_ROLL_SAMPLES else list(ring)
    collected = len(window)
    stream = capture.sample_stream(GPIO)
    while collected < WINDOW_SAMPLES:
        window.append(next(stream))
        collected += 1

    cleaned = _interpolate_gaps(window)
    if not any(v != 0.0 for v in cleaned):
        print("  Warning: window entirely empty, discarding.")
        return None

    return cleaned


def run_collection(label: str, n_sessions: int) -> None:
    """Interactive collection loop for one label."""
    _ensure_data_dir()

    try:
        import RPi.GPIO as GPIO  # type: ignore
    except ImportError:
        print("RPi.GPIO not available — running in mock mode (data will be zeros)")
        import types
        GPIO = types.SimpleNamespace(
            BCM=0, OUT=0, IN=0,
            setmode=lambda _: None,
            setup=lambda *_: None,
            output=lambda *_: None,
            input=lambda _: 0,
            cleanup=lambda: None,
        )

    capture.setup_gpio(GPIO)
    path = _session_path(label)

    try:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["timestamp_ms"] + [f"d{i}" for i in range(WINDOW_SAMPLES)] + ["label"]
            writer.writerow(header)

            for session in range(1, n_sessions + 1):
                print(f"\n[{session}/{n_sessions}]  Label: {label}")
                window = collect_one(GPIO, label)
                if window is None:
                    print("  Skipping empty window.")
                    continue

                ts_ms = int(time.time() * 1000)
                writer.writerow([ts_ms] + window + [label])
                f.flush()
                print(f"  Saved  {ts_ms}  ({len(window)} samples)")

                if session < n_sessions:
                    print("  Rest for 2 s…")
                    time.sleep(2.0)

    finally:
        GPIO.cleanup()

    print(f"\nDone. Written to {path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Gesture data collection")
    parser.add_argument("--label", choices=LABELS,
                        help="Gesture label to record")
    parser.add_argument("--sessions", type=int, default=20,
                        help="Number of gesture repetitions to capture (default 20)")
    parser.add_argument("--list-labels", action="store_true",
                        help="Print available labels and exit")
    args = parser.parse_args()

    if args.list_labels:
        print("Available labels:")
        for lbl in LABELS:
            print(f"  {lbl}")
        return

    if args.label is None:
        parser.error("--label is required unless --list-labels is set")

    run_collection(args.label, args.sessions)


if __name__ == "__main__":
    main()
