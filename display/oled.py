"""
display/oled.py
---------------
SSD1306 OLED display driver wrapper.

Uses the ``luma.oled`` library (which itself uses ``smbus2`` over I²C).

Wiring (I²C)
    SDA → GPIO 2 (pin 3)
    SCL → GPIO 3 (pin 5)
    VCC → 3.3 V
    GND → GND

The ``OLEDDisplay`` class wraps ``luma.oled`` with a simple draw-text API
and a graceful stub for non-Pi environments.
"""

from typing import Optional

try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import ssd1306
    from PIL import Image, ImageDraw, ImageFont
    _LUMA_AVAILABLE = True
except ImportError:
    _LUMA_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

OLED_WIDTH: int = 128
OLED_HEIGHT: int = 64
OLED_I2C_PORT: int = 1      # /dev/i2c-1 on modern Pi; may be 0 on old models
OLED_I2C_ADDRESS: int = 0x3C


# ── Driver ────────────────────────────────────────────────────────────────────

class OLEDDisplay:
    """
    Thin wrapper around an SSD1306 128×64 OLED.

    Falls back to stdout when ``luma.oled`` is not installed (e.g. on a
    development laptop).

    Usage::

        oled = OLEDDisplay()
        oled.show_text("Ready")
        oled.clear()
    """

    def __init__(
        self,
        port: int = OLED_I2C_PORT,
        address: int = OLED_I2C_ADDRESS,
        width: int = OLED_WIDTH,
        height: int = OLED_HEIGHT,
    ) -> None:
        self.width = width
        self.height = height
        self._device = None

        if _LUMA_AVAILABLE:
            serial = i2c(port=port, address=address)
            self._device = ssd1306(serial, width=width, height=height)
        else:
            print("[OLEDDisplay] luma.oled not available — running in stub mode.")

    def show_text(
        self,
        line1: str = "",
        line2: str = "",
        line3: str = "",
        font_size: int = 14,
    ) -> None:
        """
        Render up to three lines of text on the OLED.

        Parameters
        ----------
        line1, line2, line3:
            Text to display on each row.  Empty strings are skipped.
        font_size:
            Font size in pixels (default bitmap font is used if PIL can't
            load a TTF).
        """
        if not _LUMA_AVAILABLE:
            print(f"[OLED] {line1}  {line2}  {line3}".strip())
            return

        img = Image.new("1", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)

        try:
            from PIL import ImageFont as _IF
            font = _IF.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (IOError, ImportError):
            font = ImageFont.load_default()

        y = 0
        for line in (line1, line2, line3):
            if line:
                draw.text((0, y), line, fill=255, font=font)
                y += font_size + 2

        self._device.display(img)

    def clear(self) -> None:
        """Turn off all pixels."""
        if _LUMA_AVAILABLE and self._device is not None:
            self._device.clear()
        else:
            print("[OLED] <clear>")
