from __future__ import annotations

from gesture_detection.display.base import StatusDisplay

try:
    from luma.core.interface.serial import i2c
    from luma.core.render import canvas
    from luma.oled.device import ssd1306
except ImportError:  # pragma: no cover - exercised on Raspberry Pi hardware only.
    i2c = None
    canvas = None
    ssd1306 = None


class SSD1306Display(StatusDisplay):
    """OLED display wrapper for simple multi-line status messages."""

    def __init__(self, port: int, address: int, width: int, height: int) -> None:
        if i2c is None or canvas is None or ssd1306 is None:
            raise RuntimeError(
                "luma.oled is not installed. Install luma.oled on the Raspberry Pi to enable OLED output."
            )

        serial = i2c(port=port, address=address)
        self._device = ssd1306(serial, width=width, height=height)

    def show_lines(self, lines: list[str]) -> None:
        with canvas(self._device) as draw:
            for index, line in enumerate(lines[:4]):
                draw.text((0, index * 14), line, fill="white")
