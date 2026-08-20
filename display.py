"""
display.py — SSD1306 OLED helper for gesture readout.

Wraps luma.oled to show:
  - detected gesture name (large text)
  - confidence bar
  - idle / listening state

Hardware: SSD1306 128×64 OLED over I²C (address 0x3C by default).

Enable I²C on the Pi with:
    sudo raspi-config → Interface Options → I2C → Enable

Usage:
    from display import Display
    with Display() as disp:
        disp.show_label("swipe_left", confidence=0.92)
        time.sleep(1)
        disp.show_idle()
"""

import time
from contextlib import contextmanager
from typing import Optional

try:
    from luma.core.interface.serial import i2c
    from luma.core.render import canvas
    from luma.oled.device import ssd1306
    from PIL import ImageFont
    _LUMA_AVAILABLE = True
except ImportError:
    _LUMA_AVAILABLE = False

# ---------------------------------------------------------------------------
# OLED configuration
# ---------------------------------------------------------------------------
I2C_PORT: int = 1          # /dev/i2c-1 on most Pi models
I2C_ADDRESS: int = 0x3C
SCREEN_WIDTH: int = 128
SCREEN_HEIGHT: int = 64

FONT_LARGE_SIZE: int = 18
FONT_SMALL_SIZE: int = 10

# Confidence bar geometry
BAR_X: int = 0
BAR_Y: int = 52
BAR_W: int = 128
BAR_H: int = 8


class Display:
    """
    Context-manager wrapper around the SSD1306 OLED.

    Falls back to stdout logging when luma.oled is not installed,
    so code can be developed on a laptop without hardware.
    """

    def __init__(self) -> None:
        self._device = None
        self._font_large = None
        self._font_small = None

    def __enter__(self) -> "Display":
        self._setup()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _setup(self) -> None:
        if not _LUMA_AVAILABLE:
            print("[Display] luma.oled not available — using stdout fallback")
            return
        serial = i2c(port=I2C_PORT, address=I2C_ADDRESS)
        self._device = ssd1306(serial, width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
        try:
            self._font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                                  FONT_LARGE_SIZE)
            self._font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                                  FONT_SMALL_SIZE)
        except OSError:
            # Fall back to default bitmap font
            self._font_large = ImageFont.load_default()
            self._font_small = ImageFont.load_default()

    def close(self) -> None:
        if self._device is not None:
            self._device.cleanup()
            self._device = None

    # ------------------------------------------------------------------
    # Display states
    # ------------------------------------------------------------------

    def show_idle(self) -> None:
        """Show 'Listening…' state — sensor is active but no gesture detected."""
        if self._device is None:
            print("[Display] idle")
            return
        with canvas(self._device) as draw:
            draw.text((10, 20), "Listening...", font=self._font_small, fill="white")

    def show_label(self, label: str, confidence: float = 1.0) -> None:
        """
        Display a detected gesture name and confidence bar.

        Parameters
        ----------
        label      : gesture name string (e.g. "swipe_left")
        confidence : float in [0, 1]
        """
        if self._device is None:
            print(f"[Display] {label}  ({confidence:.0%})")
            return

        bar_fill = int(BAR_W * confidence)
        display_text = label.replace("_", " ")

        with canvas(self._device) as draw:
            # Gesture name — centred
            try:
                bbox = draw.textbbox((0, 0), display_text, font=self._font_large)
                text_w = bbox[2] - bbox[0]
            except AttributeError:
                text_w = len(display_text) * 10  # fallback estimate
            x = max(0, (SCREEN_WIDTH - text_w) // 2)
            draw.text((x, 16), display_text, font=self._font_large, fill="white")

            # Confidence percentage
            draw.text((0, 40), f"{confidence:.0%}", font=self._font_small, fill="white")

            # Confidence bar
            draw.rectangle([BAR_X, BAR_Y, BAR_X + BAR_W, BAR_Y + BAR_H], outline="white")
            if bar_fill > 0:
                draw.rectangle([BAR_X, BAR_Y, BAR_X + bar_fill, BAR_Y + BAR_H], fill="white")

    def show_message(self, line1: str, line2: str = "") -> None:
        """Generic two-line message (used for errors / boot messages)."""
        if self._device is None:
            print(f"[Display] {line1}  {line2}")
            return
        with canvas(self._device) as draw:
            draw.text((0, 10), line1, font=self._font_small, fill="white")
            if line2:
                draw.text((0, 30), line2, font=self._font_small, fill="white")


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from collect import LABELS

    with Display() as disp:
        disp.show_message("Gesture", "Detection")
        time.sleep(1.5)
        disp.show_idle()
        time.sleep(1.0)
        for lbl in LABELS:
            for conf in [0.65, 0.85, 0.97]:
                disp.show_label(lbl, confidence=conf)
                time.sleep(0.8)
