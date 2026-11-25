"""Awkward and hostile requests: the agent answers sensibly, never raises, never leaks internals."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agent import Agent
from app.schemas import UserInput
from app.trace import Tracer
from app.util import clean_label

from .invariants import check_result, check_trace
from .nasty import CASES, REJECTED


@pytest.mark.parametrize("label,fields,allowed", CASES, ids=[c[0] for c in CASES])
def test_awkward_request_gets_a_sensible_answer(settings, label, fields, allowed):
    tracer = Tracer()
    res = Agent(settings, None).run(UserInput(**fields), tracer, allow_llm=False)
    assert res.status in allowed, (res.status, res.message)
    assert check_result(res, UserInput(**fields)) == []
    assert check_trace(tracer.events) == []


@pytest.mark.parametrize("label,fields,field", REJECTED, ids=[c[0] for c in REJECTED])
def test_unusable_fields_are_refused_with_a_message_that_names_the_field(label, fields, field):
    with pytest.raises(ValidationError) as err:
        UserInput(**fields)
    assert any(field in e["loc"] for e in err.value.errors())


def test_budget_error_tells_the_person_what_to_type():
    with pytest.raises(ValidationError) as err:
        UserInput(city="Pune", budget="plenty")
    assert "for example 2000" in str(err.value)


def test_odd_text_is_not_echoed_back_as_markup(settings):
    res = Agent(settings, None).run(UserInput(city="Zürich <img src=x onerror=alert(1)>", budget=900), Tracer(), allow_llm=False)
    blob = res.model_dump_json()
    assert "<" not in blob and "onerror=" not in blob
    assert res.status == "fallback" and "Zürich" in res.message


def test_the_ambiguous_city_message_asks_instead_of_guessing(settings):
    res = Agent(settings, None).run(UserInput(city="Pune or Bangalore", budget=900), Tracer(), allow_llm=False)
    assert res.status == "needs_info" and any("more than one city" in a for a in res.assumptions)
    assert any("city" in q.lower() for q in res.questions)


def test_unknown_interests_are_listed_once_and_cleaned(settings):
    res = Agent(settings, None).run(UserInput(city="Pune", interests=["food", "Bird watching!!", "bird watching", "<b>x</b>"]), Tracer(), allow_llm=False)
    note = next(a for a in res.assumptions if "don't have venue data for" in a)
    assert note.lower().count("bird watching") == 1 and "<" not in note


@pytest.mark.parametrize("raw,expected", [
    ("  Mumbai ", "Mumbai"), ("Zürich", "Zürich"), ("New   York!!", "New York"), ("<script>x</script>", "script x script"),
    ("🏙️🌃", ""), ("12345", ""), ("", ""), (None, ""), ("a", ""), ("St. John's (north)", "St. John's (north)"), ("x" * 100, "x" * 40),
])
def test_clean_label(raw, expected):
    assert clean_label(raw) == expected
