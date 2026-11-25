"""Shared test helpers: a scripted LLM stand-in, an agent runner and a canned user input."""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tests never call OpenAI or OpenStreetMap, even if the developer has a key in .env or the shell.
os.environ["OPENAI_API_KEY"] = ""
os.environ["ENABLE_LIVE_DATA"] = "false"

from app.agent import Agent  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.llm import LLMResult  # noqa: E402
from app.schemas import UserInput  # noqa: E402
from app.trace import Tracer  # noqa: E402

ASSIGNMENT = {
    "city": "Bangalore", "budget": 2000, "available_time": "4 hours",
    "mood": "tired but wants to do something fun",
    "interests": ["food", "music", "walks"], "constraints": ["vegetarian", "avoid crowded places"],
}


class ScriptedLLM:
    """Replays canned replies per task ('parse' | 'draft' | 'final'). The last reply for a task repeats.

    A reply can be a dict (JSON body), an Exception (raised), or a callable(messages) -> dict.
    """

    name = "scripted"
    model = "scripted-1"

    def __init__(self, **script):
        self.script = {k: list(v) if isinstance(v, list) else [v] for k, v in script.items()}
        self.calls: list[tuple[str, list[dict]]] = []

    def complete_json(self, task, messages, *, effort=None, max_tokens=None):
        self.calls.append((task, messages))
        queue = self.script.get(task)
        if not queue:
            from app.llm import LLMBadOutput

            raise LLMBadOutput(f"no scripted reply for {task}")  # behaves like an unusable model reply
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(messages)
        return LLMResult(data=item, tokens=120, ms=3, model=self.model, raw=json.dumps(item))

    def count(self, task):
        return sum(1 for t, _ in self.calls if t == task)


def draft(*ids, durations=None):
    durations = durations or {}
    return {"stops": [{"id": i, "duration_min": durations.get(i), "reason": "scripted"} for i in ids], "notes": ""}


def final_for(result_ids, why="Scripted reason that fits you."):
    return {
        "headline": "A scripted Saturday", "summary": "Scripted summary.",
        "stops": [{"id": i, "why": why} for i in result_ids],
        "trade_offs": ["Scripted trade-off."], "backup": None,
    }


@pytest.fixture
def settings():
    return dataclasses.replace(load_settings(), openai_api_key=None, enable_live_data=False, max_repairs=2)


@pytest.fixture
def run(settings):
    def _run(llm=None, allow_llm=True, **user):
        tracer = Tracer()
        result = Agent(settings, llm).run(UserInput(**user), tracer, allow_llm=allow_llm)
        return result, tracer

    return _run
