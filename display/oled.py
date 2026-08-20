"""
SSD1306 OLED display driver (128 × 64, I²C).

Wraps the luma.oled library to provide a simple, high-level API used by
the main runtime.  Falls back to console output when the display or the
library is unavailable (useful for off-Pi development).

Usage
-----
    display = OLEDDisplay()
    display.show_gesture("swipe_left", confidence=0.93)
    display.show_idle()
    display.show_message("Booting…")
    display.clear()
"""

import config

try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import ssd1306
    from luma.core.render import canvas
    from PIL import ImageFont
    _LUMA_AVAILABLE = True
except ImportError:
    _LUMA_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont as PILImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_device():
    """Create the luma ssd1306 device or raise ImportError."""
    if not _LUMA_AVAILABLE:
        raise ImportError(
            "luma.oled is not installed.  "
            "Run:  pip install luma.oled"
        )
    serial = i2c(port=1, address=config.OLED_I2C_ADDRESS)
    return ssd1306(serial, width=config.OLED_WIDTH, height=config.OLED_HEIGHT)


def _default_font():
    """Return a PIL font (default bitmap if truetype unavailable)."""
    try:
        return PILImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        return PILImageFont.load_default()


def _small_font():
    try:
        return PILImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        return PILImageFont.load_default()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class OLEDDisplay:
    """
    High-level interface to the SSD1306 OLED.

    If the hardware or luma.oled library are unavailable, all methods
    fall back to printing to the console so the rest of the code runs.
    """

    def __init__(self, simulate: bool = False):
        """
        Parameters
        ----------
        simulate : If True, never attempt to open the hardware device
                   and always print to stdout instead.
        """
        self.simulate = simulate
        self._device = None

        if not simulate:
            try:
                self._device = _make_device()
            except Exception as exc:
                print(f"[OLEDDisplay] Could not open device: {exc} — falling back to console.")
                self.simulate = True

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def show_gesture(self, label: str, confidence: float) -> None:
        """
        Display the predicted gesture name and confidence percentage.

        Layout (128 × 64):
            ┌──────────────┐
            │  swipe_left  │  ← large font, centred
            │    93 %      │  ← smaller font, centred
            └──────────────┘
        """
        if self.simulate:
            print(f"[OLED] Gesture: {label}  ({confidence * 100:.0f}%)")
            return

        with canvas(self._device) as draw:
            font_large = _default_font()
            font_small = _small_font()
            w, h = config.OLED_WIDTH, config.OLED_HEIGHT

            # Gesture name
            bbox = draw.textbbox((0, 0), label, font=font_large)
            text_w = bbox[2] - bbox[0]
            draw.text(((w - text_w) // 2, 8), label, font=font_large, fill="white")

            # Confidence
            conf_str = f"{confidence * 100:.0f}%"
            bbox2 = draw.textbbox((0, 0), conf_str, font=font_small)
            text_w2 = bbox2[2] - bbox2[0]
            draw.text(((w - text_w2) // 2, 38), conf_str, font=font_small, fill="white")

    def show_idle(self) -> None:
        """Display a low-key idle indicator (e.g. a dot or blank screen)."""
        if self.simulate:
            print("[OLED] (idle)")
            return

        with canvas(self._device) as draw:
            # Minimal idle indicator: small dot in the centre
            cx, cy = config.OLED_WIDTH // 2, config.OLED_HEIGHT // 2
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill="white")

    def show_message(self, text: str) -> None:
        """Display an arbitrary status message (boot, error, etc.)."""
        if self.simulate:
            print(f"[OLED] {text}")
            return

        with canvas(self._device) as draw:
            font = _small_font()
            draw.text((4, 24), text, font=font, fill="white")

    def clear(self) -> None:
        """Blank the display."""
        if self.simulate:
            print("[OLED] (clear)")
            return

        if self._device:
            self._device.clear()

    def __del__(self):
        self.clear()
