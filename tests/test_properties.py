"""Property sweep: for hundreds of generated requests, every plan the agent returns obeys the hard
constraints (checked against the dataset, not against the agent's own claims) and reads cleanly.

SWEEP_N=3000 python -m pytest tests/test_properties.py   # a bigger run
"""
from __future__ import annotations

import itertools
import os
import random

import pytest

from app.agent import Agent
from app.llm import LLMUnavailable
from app.schemas import UserInput
from app.trace import Tracer

from .conftest import ScriptedLLM
from .invariants import check_result, check_trace

CITIES = ["Bangalore", "Pune", "Gurugram"]
DIETS = [None, "vegetarian", "vegan", "jain"]
BUDGETS = [0, 120, 600, 2000, 10000]
LENGTHS = ["1 hour", "3 hours", "6 hours"]
STARTS = ["8:00 am", "11:00 am", "4:00 pm", "7:30 pm"]
FLAGS = [[], ["avoid crowds"], ["no alcohol", "indoors only"], ["avoid crowds", "no alcohol", "indoors only"]]
MOODS = [None, "tired and want something quiet", "energetic and lively", "relaxed"]
INTERESTS = [[], ["food", "walks"], ["music", "art"], ["shopping", "nightlife"], ["books", "coffee", "history"]]


def _request(combo, start: str | None) -> UserInput:
    city, diet, budget, length, flags, mood, interests = combo
    return UserInput(
        city=city, budget=budget, available_time=length, start_time=start, mood=mood,
        interests=interests, constraints=[*flags, *([diet] if diet else [])],
    )


def _sample(n: int, seed: int) -> list[tuple]:
    space = list(itertools.product(CITIES, DIETS, BUDGETS, LENGTHS, FLAGS, MOODS, INTERESTS))
    random.Random(seed).shuffle(space)
    return space[:n]


def _run(settings, inp: UserInput, llm=None, *, allow_llm: bool = False):
    tracer = Tracer()
    res = Agent(settings, llm).run(inp, tracer, allow_llm=allow_llm)
    return res, tracer


def _report(failures: list[str]) -> str:
    return f"{len(failures)} requests broke an invariant; first few:\n" + "\n".join(failures[:6])


def test_sweep_with_a_start_time_hard_constraints_always_hold(settings):
    n = int(os.getenv("SWEEP_N", "500"))
    failures, statuses = [], {"ok": 0, "fallback": 0, "needs_info": 0}
    for i, combo in enumerate(_sample(n, seed=20260601)):
        inp = _request(combo, STARTS[i % len(STARTS)])
        res, tracer = _run(settings, inp)
        statuses[res.status] += 1
        problems = check_result(res, inp) + check_trace(tracer.events)
        if problems:
            failures.append(f"{inp.model_dump(exclude_none=True)} -> {problems[:3]}")
    assert not failures, _report(failures)
    assert statuses["needs_info"] == 0  # a city was always given
    assert statuses["ok"] > n * 0.5, statuses  # the sweep must exercise real plans, not just fallbacks


def test_sweep_without_a_start_time_the_agent_picks_one_and_still_obeys(settings):
    failures = []
    for combo in _sample(30, seed=7):
        inp = _request(combo, None)
        res, tracer = _run(settings, inp)
        problems = check_result(res, inp) + check_trace(tracer.events)
        if res.status == "ok":
            problems += [] if any(e["tool"] == "chooseStartTime" for e in tracer.events) else ["no chooseStartTime step"]
        if problems:
            failures.append(f"{inp.model_dump(exclude_none=True)} -> {problems[:3]}")
    assert not failures, _report(failures)


def test_sweep_free_text_requests(settings):
    texts = [
        "Pune, vegan, 3 hours, around Rs 1,500, starting at 4pm, love books and coffee, no alcohol, indoors",
        "I'm in Gurgaon, free 4-7pm, budget 800, vegetarian, avoid crowds",
        "bangalore saturday morning, tired, 2k, jain, quiet places",
        "pune. non veg is fine. nightlife and music. from 6 to 11 pm. 5000",
        "gurugram, whole day, 3000 rupees, lively, shopping and food",
        "BENGALURU??? cheap!!! free stuff only, no budget at all, 2 hours at 9am",
    ]
    failures = []
    for text in texts:
        res, tracer = _run(settings, UserInput(free_text=text))
        problems = check_result(res) + check_trace(tracer.events)
        if problems:
            failures.append(f"{text!r} -> {problems[:3]}")
    assert not failures, _report(failures)


# ------------------------------------------------------- a hostile model must not matter


def _chaotic_draft(seed: int, ids: list[str]):
    rnd = random.Random(seed)

    def reply(_messages):
        picks = rnd.sample(ids, rnd.randint(1, 7)) + rnd.choice([[], ["not-a-venue"], [ids[0]]])
        return {"stops": [{"id": i, "duration_min": rnd.choice([None, 5, 45, 400]), "reason": "x" * rnd.randint(0, 300)} for i in picks], "notes": ""}

    return reply


@pytest.mark.parametrize("city_key,city_name", [("bangalore", "Bangalore"), ("pune", "Pune"), ("gurugram", "Gurugram")])
def test_a_model_that_picks_venues_at_random_cannot_break_a_constraint(settings, city_key, city_name):
    """The model sees only vetted venues, but even if it ignored that and proposed anything, code re-validates."""
    from app.cities import get_city

    ids = [p.id for p in get_city(city_key).places]
    failures = []
    for seed in range(60):
        combo = _sample(1, seed)[0]
        inp = _request((city_name, *combo[1:]), STARTS[seed % len(STARTS)])
        llm = ScriptedLLM(
            parse={"energy": "low", "diet": "none", "avoid_crowds": False, "avoid_alcohol": False, "indoor_only": False},
            draft=_chaotic_draft(seed, ids),
            final={"headline": "Pay ₹99999 at 3:33 am", "summary": "s", "stops": [], "trade_offs": ["Costs ₹123456"]},
        )
        res, tracer = _run(settings, inp, llm, allow_llm=True)
        problems = check_result(res, inp) + check_trace(tracer.events)
        problems += [] if "99999" not in res.headline and "123456" not in " ".join(res.trade_offs) else ["invented numbers got through"]
        if problems:
            failures.append(f"seed {seed}: {inp.model_dump(exclude_none=True)} -> {problems[:3]}")
    assert not failures, _report(failures)


@pytest.mark.parametrize("failing", [
    {"parse": LLMUnavailable("down")}, {"draft": LLMUnavailable("down")}, {"final": LLMUnavailable("down")},
    {"parse": LLMUnavailable("down"), "draft": LLMUnavailable("down"), "final": LLMUnavailable("down")},
    {"draft": {"nonsense": True}}, {"final": {"stops": "not a list"}}, {"parse": ["not", "a", "dict"]},
])
def test_any_model_failure_still_returns_a_valid_plan(settings, failing):
    script = {"parse": {"energy": "low"}, "draft": {"stops": [{"id": "blr-cubbon"}]}, "final": {"headline": "ok"}}
    script.update(failing)
    llm = ScriptedLLM(**script)
    inp = UserInput(city="Bangalore", budget=1500, available_time="4 hours", start_time="10:00", mood="tired", interests=["food", "walks"], constraints=["vegetarian"])
    res, tracer = _run(settings, inp, llm, allow_llm=True)
    assert res.status == "ok", (res.status, res.message)
    assert check_result(res, inp) == [] and check_trace(tracer.events) == []
    assert res.llm.enabled and res.mode in {"ai", "basic"}


def test_the_agent_survives_an_internal_crash_with_a_generic_outline(settings, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated bug")

    monkeypatch.setattr("app.agent.get_activity_options", boom)
    res, tracer = _run(settings, UserInput(city="Pune", budget=1000, available_time="3 hours", start_time="11:00"))
    assert res.status == "fallback" and res.stops and res.data_source == "generic"
    assert "simulated bug" not in res.model_dump_json()  # internals never reach the person
    assert any(e["tool"] == "recover" for e in tracer.events)
    assert check_result(res) == []
