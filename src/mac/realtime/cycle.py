from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TenSecondCycleClock:
    cycle_ms: int = 10_000
    condition: str | None = None
    origin_ms: int | None = None
    cycle_index: int = 0
    last_message_ms: int | None = None

    def condition_event(self, condition: str, unix_time_ms: int) -> bool:
        if self.last_message_ms is not None and unix_time_ms < self.last_message_ms:
            return False
        self.last_message_ms = unix_time_ms
        changed = condition != self.condition
        if changed:
            self.condition = condition
            self.origin_ms = unix_time_ms
            self.cycle_index = 0
        return changed

    def initialize(self, unix_time_ms: int, condition: str | None = None) -> None:
        self.condition = condition
        self.origin_ms = (unix_time_ms // self.cycle_ms) * self.cycle_ms
        self.cycle_index = 0

    def ready_windows(self, unix_time_ms: int) -> list[tuple[int, int, int]]:
        if self.origin_ms is None:
            self.initialize(unix_time_ms, self.condition)
        output = []
        while unix_time_ms >= self.origin_ms + (self.cycle_index + 1) * self.cycle_ms:
            start = self.origin_ms + self.cycle_index * self.cycle_ms
            output.append((self.cycle_index, start, start + self.cycle_ms))
            self.cycle_index += 1
        return output


class TimeBuffer:
    def __init__(self) -> None:
        self.rows: list[tuple[int, object]] = []

    def add(self, unix_time_ms: int, value: object) -> None:
        self.rows.append((int(unix_time_ms), value))

    def window(self, start_ms: int, end_ms: int) -> list[object]:
        selected = [value for timestamp, value in self.rows if start_ms <= timestamp < end_ms]
        self.rows = [(timestamp, value) for timestamp, value in self.rows if timestamp >= end_ms]
        return selected

    def clear(self) -> None:
        self.rows.clear()

