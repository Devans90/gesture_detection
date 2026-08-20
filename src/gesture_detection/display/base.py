from __future__ import annotations

from abc import ABC, abstractmethod


class StatusDisplay(ABC):
    @abstractmethod
    def show_lines(self, lines: list[str]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Release display resources if needed."""
