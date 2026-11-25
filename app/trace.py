"""The agent trace: every tool call becomes an event, optionally streamed as it happens."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


@dataclass
class ToolInfo:
    """Side-channel a tool uses to report what it did (LLM calls, retries, fallbacks)."""

    llm_calls: int = 0
    tokens: int = 0
    retries: int = 0
    fallback: str | None = None
    unavailable: bool = False  # the model itself is unusable (auth, quota, outage), not just one bad reply
    notes: list[str] = field(default_factory=list)


class StepHandle:
    """Mutable record the orchestrator fills in while a step is running."""

    def __init__(self) -> None:
        self.status: str = "ok"  # ok | retry | fallback | error | info
        self.summary: str = ""
        self.detail: Any = None
        self.label: str | None = None


class Tracer:
    def __init__(self, on_event: Callable[[dict], None] | None = None) -> None:
        self.events: list[dict] = []
        self._on = on_event
        self._seq = 0

    def _emit(self, ev: dict) -> None:
        if self._on is not None:
            try:
                self._on(ev)
            except Exception:  # a broken consumer must never break the agent
                pass

    @contextmanager
    def step(self, tool: str, label: str | None = None, inp: str = "") -> Iterator[StepHandle]:
        self._seq += 1
        rec: dict = {
            "type": "step",
            "id": self._seq,
            "tool": tool,
            "label": label or tool,
            "state": "running",
            "input": inp,
            "summary": "",
            "ms": 0,
            "detail": None,
        }
        self._emit(dict(rec))
        handle = StepHandle()
        t0 = time.perf_counter()
        try:
            yield handle
            rec["state"] = handle.status
        except Exception as e:  # noqa: BLE001 - recorded, then re-raised
            rec["state"] = "error"
            handle.summary = handle.summary or f"{type(e).__name__}: {e}"
            raise
        finally:
            rec["summary"] = handle.summary
            rec["detail"] = handle.detail
            if handle.label:
                rec["label"] = handle.label
            rec["ms"] = int((time.perf_counter() - t0) * 1000)
            self.events.append(dict(rec))
            self._emit(dict(rec))

    def note(self, tool: str, summary: str, status: str = "info", label: str | None = None) -> None:
        """An instantaneous event (no duration), e.g. 'relaxed a constraint'."""
        with self.step(tool, label=label) as st:
            st.status = status
            st.summary = summary
