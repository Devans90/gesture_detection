from __future__ import annotations

from gesture_detection.display.base import StatusDisplay


class ConsoleDisplay(StatusDisplay):
    def __init__(self) -> None:
        self._previous: tuple[str, ...] = ()

    def show_lines(self, lines: list[str]) -> None:
        current = tuple(lines)
        if current == self._previous:
            return
        self._previous = current
        print("\n".join(lines))
        print("-" * 32)
