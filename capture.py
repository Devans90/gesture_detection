"""
capture.py — HC-SR04 distance sampling for Raspberry Pi.

Samples the ultrasonic sensor continuously and yields cleaned distance
readings in centimetres.  Handles:
  - GPIO trigger/echo timing
  - dropped readings (echo never arrives)
  - out-of-range values (< MIN_CM or > MAX_CM)
  - optional median filter to suppress spike noise

Usage (standalone test):
    python capture.py
"""

import time
from collections import deque
from typing import Generator, Optional

# ---------------------------------------------------------------------------
# Pin configuration — adjust to your wiring
# ---------------------------------------------------------------------------
TRIG_PIN: int = 23   # BCM numbering
ECHO_PIN: int = 24

# ---------------------------------------------------------------------------
# Sensor limits
# ---------------------------------------------------------------------------
MIN_CM: float = 2.0    # HC-SR04 spec minimum reliable range
MAX_CM: float = 400.0  # HC-SR04 spec maximum reliable range
SAMPLE_RATE_HZ: float = 20.0  # HC-SR04 practical max; set lower if needed
ECHO_TIMEOUT_S: float = 0.05  # seconds before declaring a dropped reading

# ---------------------------------------------------------------------------
# Noise filter
# ---------------------------------------------------------------------------
MEDIAN_WINDOW: int = 3  # set to 1 to disable


def _median(values: list) -> float:
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def _trigger_pulse(GPIO) -> None:
    """Send a 10 µs trigger pulse."""
    GPIO.output(TRIG_PIN, True)
    time.sleep(10e-6)
    GPIO.output(TRIG_PIN, False)


def _measure_echo(GPIO) -> Optional[float]:
    """
    Wait for the echo pulse and convert its duration to centimetres.

    Returns None if the echo never arrives within ECHO_TIMEOUT_S.
    """
    deadline = time.monotonic() + ECHO_TIMEOUT_S

    # Wait for echo to go high
    while GPIO.input(ECHO_PIN) == 0:
        if time.monotonic() > deadline:
            return None
    start = time.monotonic()

    # Wait for echo to go low
    while GPIO.input(ECHO_PIN) == 1:
        if time.monotonic() > deadline:
            return None
    end = time.monotonic()

    distance_cm = (end - start) * 34300.0 / 2.0  # speed of sound, round-trip
    return distance_cm


def setup_gpio(GPIO) -> None:
    """Initialise GPIO pins for HC-SR04.  Call once at startup."""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.output(TRIG_PIN, False)
    time.sleep(0.05)  # let sensor settle


def read_distance(GPIO) -> Optional[float]:
    """
    Take a single distance measurement.

    Returns distance in cm, or None if the reading is dropped or out of range.
    """
    _trigger_pulse(GPIO)
    raw = _measure_echo(GPIO)

    if raw is None:
        return None
    if not (MIN_CM <= raw <= MAX_CM):
        return None
    return raw


def sample_stream(GPIO) -> Generator[Optional[float], None, None]:
    """
    Infinite generator of distance readings at approximately SAMPLE_RATE_HZ.

    Yields cleaned, median-filtered readings (cm) or None for dropped frames.
    Caller is responsible for GPIO setup/cleanup.
    """
    interval = 1.0 / SAMPLE_RATE_HZ
    window: deque = deque(maxlen=MEDIAN_WINDOW)

    while True:
        t0 = time.monotonic()

        raw = read_distance(GPIO)
        if raw is not None:
            window.append(raw)
            value: Optional[float] = _median(list(window))
        else:
            value = None

        yield value

        elapsed = time.monotonic() - t0
        sleep_s = max(0.0, interval - elapsed)
        time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        import RPi.GPIO as GPIO  # type: ignore
    except ImportError:
        print("RPi.GPIO not available — running in mock mode")

        class _MockGPIO:  # pragma: no cover
            BCM = OUT = IN = 0

            @staticmethod
            def setmode(_): pass

            @staticmethod
            def setup(_, __): pass

            @staticmethod
            def output(_, __): pass

            @staticmethod
            def input(_): return 0

            @staticmethod
            def cleanup(): pass

        GPIO = _MockGPIO()

    setup_gpio(GPIO)
    print(f"Sampling at {SAMPLE_RATE_HZ} Hz  (Ctrl-C to stop)\n")
    try:
        for i, dist in enumerate(sample_stream(GPIO)):
            if dist is None:
                print(f"[{i:6d}] -- dropped --")
            else:
                bar = "#" * int(dist / 5)
                print(f"[{i:6d}] {dist:6.1f} cm  {bar}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        GPIO.cleanup()
