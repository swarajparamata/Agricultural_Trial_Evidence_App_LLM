"""Collects the agent's step events and forwards them to a live listener (the SSE stream)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .schemas import AgentEvent, LogStatus

EventListener = Callable[[AgentEvent], None]


class EventLog:
    def __init__(self, listener: EventListener | None = None) -> None:
        self._started = time.perf_counter()
        self._listener = listener
        self.events: list[AgentEvent] = []

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._started) * 1000)

    def log(self, message: str, status: LogStatus = "info", *, node: str | None = None, **data: Any) -> AgentEvent:
        event = AgentEvent(
            type="log", status=status, message=message, node=node, elapsed_ms=self.elapsed_ms(), data=data or None
        )
        self.events.append(event)
        if self._listener is not None:
            self._listener(event)
        return event
