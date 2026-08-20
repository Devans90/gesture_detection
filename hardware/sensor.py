"""
HC-SR04 ultrasonic sensor driver.

Handles the trigger/echo cycle, converts pulse width to centimetres,
and clamps out-of-range readings.

Usage (on a real Pi with RPi.GPIO installed):
    sensor = HCSR04()
    distance = sensor.read_cm()
    sensor.cleanup()

In tests or on non-Pi hardware this module can be imported normally;
the class will raise RuntimeError if RPi.GPIO is unavailable and
``simulate=False`` (the default).
"""

import time

import config

try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False


class HCSR04:
    """Low-level driver for the HC-SR04 ultrasonic distance sensor."""

    # Speed of sound at ~20 °C in cm/µs
    SOUND_SPEED_CM_PER_US = 0.03432

    def __init__(
        self,
        trig_pin: int = config.TRIG_PIN,
        echo_pin: int = config.ECHO_PIN,
        simulate: bool = False,
    ):
        """
        Parameters
        ----------
        trig_pin:  BCM GPIO pin connected to the HC-SR04 TRIG line.
        echo_pin:  BCM GPIO pin connected to the HC-SR04 ECHO line.
        simulate:  If True, return synthetic readings (useful for off-Pi dev).
        """
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.simulate = simulate

        if not simulate:
            if not _GPIO_AVAILABLE:
                raise RuntimeError(
                    "RPi.GPIO is not installed. "
                    "Run on a Raspberry Pi or pass simulate=True."
                )
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.trig_pin, GPIO.OUT)
            GPIO.setup(self.echo_pin, GPIO.IN)
            # Ensure trigger is low before we start
            GPIO.output(self.trig_pin, GPIO.LOW)
            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_cm(self) -> float | None:
        """
        Take a single distance measurement.

        Returns
        -------
        Distance in centimetres, or None if the reading is out of the
        sensor's reliable range or a timeout occurs.
        """
        if self.simulate:
            return self._simulate_reading()
        return self._hardware_reading()

    def cleanup(self) -> None:
        """Release GPIO resources.  Call this when you are done."""
        if not self.simulate and _GPIO_AVAILABLE:
            GPIO.cleanup()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hardware_reading(self) -> float | None:
        """Trigger the sensor and measure the echo pulse width."""
        # Send 10 µs trigger pulse
        GPIO.output(self.trig_pin, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, GPIO.LOW)

        timeout = time.monotonic() + 0.04  # 40 ms overall timeout

        # Wait for echo to go HIGH
        while GPIO.input(self.echo_pin) == GPIO.LOW:
            if time.monotonic() > timeout:
                return None  # no echo received
        pulse_start = time.monotonic()

        # Wait for echo to go LOW
        while GPIO.input(self.echo_pin) == GPIO.HIGH:
            if time.monotonic() > timeout:
                return None  # echo never fell — object too close or absent
        pulse_end = time.monotonic()

        pulse_duration_us = (pulse_end - pulse_start) * 1_000_000
        distance_cm = (pulse_duration_us * self.SOUND_SPEED_CM_PER_US) / 2.0
        return self._clamp(distance_cm)

    def _simulate_reading(self) -> float:
        """
        Return a synthetic reading for offline development.

        Replace or extend this for richer simulation scenarios.
        """
        import random
        return round(random.uniform(5.0, 35.0), 2)

    @staticmethod
    def _clamp(distance_cm: float) -> float | None:
        """Return None if the reading is outside the reliable range."""
        if distance_cm < config.MIN_DISTANCE_CM:
            return None
        if distance_cm > config.MAX_DISTANCE_CM:
            return None
        return round(distance_cm, 2)
