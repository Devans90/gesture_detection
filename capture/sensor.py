"""
capture/sensor.py
-----------------
HC-SR04 ultrasonic sensor driver.

Hardware wiring
    TRIG  → GPIO 23 (BCM)
    ECHO  → GPIO 24 (BCM)   ← 5 V logic; use a voltage divider or level shifter

The HC-SR04 datasheet says:
    1. Pull TRIG high for ≥10 µs to start a measurement.
    2. Measure the ECHO pulse width; distance = pulse_width * 17150 cm/s.

Typical range: 2 cm – 400 cm.  Readings outside that window are treated as
drops (None) so the rest of the pipeline can handle them gracefully.
"""

import time
from typing import Optional

# RPi.GPIO is only available on the Pi.  Other modules import from here, so we
# provide a lightweight stub that lets the code import on a laptop.
try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    GPIO = None      # type: ignore[assignment]
    _GPIO_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────

TRIG_PIN: int = 23          # BCM numbering
ECHO_PIN: int = 24
SAMPLE_RATE_HZ: float = 20  # HC-SR04 max ~20 Hz; higher risks echo overlap
MIN_DISTANCE_CM: float = 2.0
MAX_DISTANCE_CM: float = 400.0
ECHO_TIMEOUT_S: float = 0.03  # ~5 m round-trip; bail if echo never arrives


# ── Sensor class ─────────────────────────────────────────────────────────────

class HC_SR04:
    """
    Manages a single HC-SR04 sensor.

    Usage::

        sensor = HC_SR04()
        sensor.setup()
        try:
            while True:
                d = sensor.read()
                print(d)
                time.sleep(1 / SAMPLE_RATE_HZ)
        finally:
            sensor.teardown()

    Returns ``None`` for any dropped, out-of-range, or timed-out reading.
    """

    def __init__(
        self,
        trig_pin: int = TRIG_PIN,
        echo_pin: int = ECHO_PIN,
        min_distance: float = MIN_DISTANCE_CM,
        max_distance: float = MAX_DISTANCE_CM,
        timeout: float = ECHO_TIMEOUT_S,
    ) -> None:
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.timeout = timeout

    def setup(self) -> None:
        """Initialise GPIO pins.  Call once before reading."""
        if not _GPIO_AVAILABLE:
            return
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        GPIO.output(self.trig_pin, False)
        time.sleep(0.02)  # let sensor settle

    def teardown(self) -> None:
        """Release GPIO resources."""
        if not _GPIO_AVAILABLE:
            return
        GPIO.cleanup()

    def read(self) -> Optional[float]:
        """
        Fire one ultrasonic burst and return the distance in centimetres.

        Returns
        -------
        float
            Distance in cm, or ``None`` if the reading was dropped / out of
            range / timed out.
        """
        if not _GPIO_AVAILABLE:
            # Running on a laptop — return None so callers handle it safely.
            return None

        # ── Trigger pulse ────────────────────────────────────────────────────
        GPIO.output(self.trig_pin, True)
        time.sleep(0.00001)  # 10 µs
        GPIO.output(self.trig_pin, False)

        # ── Wait for echo to go high ─────────────────────────────────────────
        pulse_start = time.time()
        deadline = pulse_start + self.timeout
        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
            if pulse_start > deadline:
                return None  # echo never arrived

        # ── Wait for echo to go low ──────────────────────────────────────────
        pulse_end = time.time()
        deadline = pulse_end + self.timeout
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()
            if pulse_end > deadline:
                return None  # echo never fell

        # ── Convert to distance ──────────────────────────────────────────────
        duration = pulse_end - pulse_start
        distance = duration * 17150  # cm = s * (34300 cm/s / 2)

        if distance < self.min_distance or distance > self.max_distance:
            return None

        return round(distance, 1)
