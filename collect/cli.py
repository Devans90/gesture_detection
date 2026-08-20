"""
collect/cli.py
--------------
Interactive command-line tool for collecting labelled gesture examples.

Run on the Pi::

    python -m collect.cli --label swipe_left --participant alice --n 30

The script:
    1. Arms the sensor and prints a countdown.
    2. Waits for the ``GestureSegmenter`` to fire (i.e. motion detected).
    3. Saves the window via ``Recorder``.
    4. Repeats until ``--n`` examples have been collected.

You don't need to change this file — it is purely scaffolding.
The gesture detection quality depends on ``_motion_detected`` in
``capture/buffer.py``.
"""

import argparse
import signal
import sys
import time

import numpy as np

from capture.sensor import HC_SR04, SAMPLE_RATE_HZ
from capture.buffer import RollingBuffer, GestureSegmenter, WINDOW_SIZE
from collect.recorder import Recorder


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gesture data collection tool")
    p.add_argument("--label", required=True, help="Gesture label, e.g. swipe_left")
    p.add_argument("--participant", default="unknown", help="Participant name or ID")
    p.add_argument("--n", type=int, default=30, help="Number of examples to collect")
    p.add_argument("--data-dir", default="data/raw", help="Root data directory")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print captured windows to stdout instead of saving to disk",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    recorder = Recorder(label=args.label, data_dir=args.data_dir)
    if not args.dry_run:
        recorder.start_session(participant=args.participant)

    collected: list[np.ndarray] = []

    def _on_gesture(window: np.ndarray) -> None:
        if len(collected) >= args.n:
            return

        collected.append(window)
        print(f"  ✓ captured example {len(collected)}/{args.n}")

        if args.dry_run:
            print(f"    window mean={window.mean():.1f} cm  std={window.std():.1f} cm")
        else:
            path = recorder.record(window)
            print(f"    saved → {path}")

    sensor = HC_SR04()
    sensor.setup()

    buffer = RollingBuffer(maxlen=WINDOW_SIZE)
    segmenter = GestureSegmenter(buffer=buffer, on_gesture=_on_gesture)

    # ── Graceful shutdown on Ctrl-C ───────────────────────────────────────────

    def _shutdown(signum, frame):  # noqa: ANN001
        print("\n[CLI] Interrupted — saving session…")
        _finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    def _finish() -> None:
        sensor.teardown()
        if not args.dry_run:
            recorder.finish_session()

    # ── Collection loop ───────────────────────────────────────────────────────

    print(f"\n[CLI] Ready.  Collecting {args.n} × '{args.label}' gestures.")
    print("      Make a gesture when you see the prompt.\n")

    interval = 1.0 / SAMPLE_RATE_HZ

    while len(collected) < args.n:
        print(f"  → Perform gesture now  [{len(collected)+1}/{args.n}]  ", end="", flush=True)

        # Wait for a window's worth of motion
        start = time.time()
        while len(collected) == len(collected):  # loop until segmenter fires
            distance = sensor.read()
            buffer.push(distance)
            try:
                segmenter.update()
            except NotImplementedError:
                print(
                    "\n[CLI] _motion_detected is not implemented yet.\n"
                    "      Open capture/buffer.py and fill in that method first."
                )
                _finish()
                sys.exit(1)

            if len(collected) > sum(1 for _ in []) or time.time() - start > 5:
                # collected count changed or 5 s timeout — break inner loop
                break
            time.sleep(interval)

    _finish()
    print(f"\n[CLI] Done. {len(collected)} examples collected.")


if __name__ == "__main__":
    main()
