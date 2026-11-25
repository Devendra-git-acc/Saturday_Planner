"""Parsing and formatting helpers: the first line of defence against messy human input."""
from __future__ import annotations

import pytest

from app.schemas import UserInput
from app.util import (
    fmt12, fmt_inr, fmt_minutes, haversine_km, overlap_minutes, parse_clock, parse_duration_minutes,
    parse_money, parse_windows, plural, short, to_number,
)


@pytest.mark.parametrize("text,expected", [
    ("4 hours", 240), ("4 hrs", 240), ("4h", 240), ("2.5 hours", 150), ("90 minutes", 90), ("45 min", 45),
    ("1 hour 30 minutes", 90), ("an hour", 60), ("an hour and a half", 90), ("half day", 300), ("half a day", 300),
    ("full day", 480), ("all day", 480), ("a couple of hours", 120), ("a few hours", 180), ("2-3 hrs", 150),
    ("3", 180), ("240", 240), ("Around 4 Hours please", 240),
])
def test_parse_duration(text, expected):
    assert parse_duration_minutes(text) == expected


@pytest.mark.parametrize("text", [None, "", "   ", "soon", "whenever", "0", "1000000", "5000"])
def test_parse_duration_rejects_nonsense(text):
    assert parse_duration_minutes(text) is None


@pytest.mark.parametrize("text,expected", [
    ("4 pm", 960), ("4:30 pm", 990), ("16:30", 990), ("9 am", 540), ("12 am", 0), ("12 pm", 720), ("12:00 pm", 720),
    ("7 pm", 1140), ("noon", 720), ("evening", 1020), ("morning", 540), ("5", 17 * 60), ("10", 600), ("9.30 am", 570),
])
def test_parse_clock(text, expected):
    assert parse_clock(text) == expected


@pytest.mark.parametrize("text", [None, "", "25:99", "banana", "99 pm"])
def test_parse_clock_rejects_nonsense(text):
    assert parse_clock(text) is None


@pytest.mark.parametrize("text,expected", [
    # the time of day is found even when other numbers come first in the sentence
    ("3 hours, starting at 4pm", 960),
    ("Rs 1,500, starting at 4pm", 960),
    ("I'm in Pune, vegan, 3 hours, around Rs 1,500, starting at 4pm, love books", 960),
    ("weekend with 6 friends, 4pm", 960),
    ("I have 2 free hours after 5", 17 * 60),
    # a bare number needs a cue word, and small numbers read as pm
    ("leave at 9", 540), ("starting at 4", 960), ("from 5 to 8", 17 * 60), ("at 4 o'clock", 960),
    ("I want to start at 10 and be home by 2", 600),
    # ranges: the first number borrows the second one's am/pm unless that would end before it starts
    ("4-7pm", 960), ("4 to 7 pm", 960), ("10-12pm", 600), ("9-5pm", 540), ("11am-2pm", 660),
])
def test_parse_clock_finds_the_time_among_other_numbers(text, expected):
    assert parse_clock(text) == expected


@pytest.mark.parametrize("text", [
    "3 hours", "2.5 hours", "1h30m", "30 min", "90 mins", "2 km away", "5 people", "5-6 people", "2-3 hours",
    "around Rs 1,500", "around 1,500 rupees", "around 1.5k", "budget 2000", "Rs 4.50 coffee", "be home by 9",
])
def test_parse_clock_ignores_quantities_money_and_end_times(text):
    assert parse_clock(text) is None


@pytest.mark.parametrize("text,expected", [
    ("around Rs 1500", 1500), ("under ₹2,000", 2000), ("2k", 2000), ("budget of 800", 800), ("1.5k", 1500),
    ("rupees 750", 750), ("2 lakh", 200000), ("INR 999", 999), ("1,200 rs", 1200),
])
def test_parse_money(text, expected):
    assert parse_money(text) == expected


def test_parse_money_ignores_other_numbers():
    assert parse_money("3 hours in Pune at 5 pm") is None
    assert parse_money("") is None and parse_money(None) is None


@pytest.mark.parametrize("value,expected", [
    (2000, 2000.0), ("2000", 2000.0), ("2,000", 2000.0), ("₹2000", 2000.0), ("2k", 2000.0), ("1 lakh", 100000.0),
    (None, None), ("", None), ("any", None), ("flexible", None), ("no limit", None), (True, None),
])
def test_to_number(value, expected):
    assert to_number(value) == expected


def test_to_number_rejects_words():
    with pytest.raises(ValueError):
        to_number("a lot")


@pytest.mark.parametrize("n,expected", [
    (0, "₹0"), (999, "₹999"), (1000, "₹1,000"), (2000, "₹2,000"), (12345, "₹12,345"), (125000, "₹1,25,000"),
    (10000000, "₹1,00,00,000"), (-1500, "-₹1,500"), (1999.6, "₹2,000"),
])
def test_fmt_inr_uses_indian_grouping(n, expected):
    assert fmt_inr(n) == expected


@pytest.mark.parametrize("minutes,expected", [(0, "0 min"), (45, "45 min"), (60, "1 h"), (75, "1 h 15 min"), (240, "4 h")])
def test_fmt_minutes(minutes, expected):
    assert fmt_minutes(minutes) == expected


@pytest.mark.parametrize("minute,expected", [
    (0, "12:00 am"), (60, "1:00 am"), (600, "10:00 am"), (720, "12:00 pm"), (13 * 60 + 5, "1:05 pm"), (1439, "11:59 pm"), (1440, "12:00 am"),
])
def test_fmt12_is_lowercase_and_wraps(minute, expected):
    assert fmt12(minute) == expected


def test_plural():
    assert plural(1, "stop") == "1 stop"
    assert plural(0, "stop") == "0 stops"
    assert plural(2, "activity", "activities") == "2 activities"
    assert plural(1, "activity", "activities") == "1 activity"


def test_windows_and_overlap():
    assert parse_windows("07:00-11:30,15:00-20:00") == ((420, 690), (900, 1200))
    assert parse_windows("") == ()
    assert overlap_minutes(0, 60, 30, 90) == 30
    assert overlap_minutes(0, 30, 30, 90) == 0
    assert overlap_minutes(0, 100, 20, 40) == 20


def test_haversine_known_distance():
    # Bangalore (MG Road) to Pune (Shivajinagar) is roughly 740 km as the crow flies.
    km = haversine_km(12.9756, 77.6068, 18.5314, 73.8446)
    assert 700 < km < 780
    assert haversine_km(12.9, 77.6, 12.9, 77.6) == 0


def test_short_truncates_and_flattens_newlines():
    assert short("a\nb") == "a b"
    assert len(short("x" * 500, 50)) == 50 and short("x" * 500, 50).endswith("…")


class TestUserInputIsLenient:
    def test_accepts_the_assignment_json(self):
        u = UserInput(**{
            "city": "Bangalore", "budget": 2000, "available_time": "4 hours", "mood": "tired but wants to do something fun",
            "interests": ["food", "music", "walks"], "constraints": ["vegetarian", "avoid crowded places"],
        })
        assert u.budget == 2000 and u.interests == ["food", "music", "walks"]

    def test_coerces_strings_numbers_and_comma_lists(self):
        u = UserInput(city=" Pune ", budget="₹1,500", available_time=4, interests="food, music;walks", constraints=None)
        assert u.city == "Pune" and u.budget == 1500 and u.available_time == "4"
        assert u.interests == ["food", "music", "walks"] and u.constraints == []

    def test_blank_strings_become_none_and_unknown_keys_are_ignored(self):
        u = UserInput(city="   ", mood="", surprise="ignored")
        assert u.city is None and u.mood is None

    def test_rejects_negative_budget_and_absurd_lengths(self):
        with pytest.raises(ValueError):
            UserInput(budget=-1)
        with pytest.raises(ValueError):
            UserInput(free_text="x" * 5000)
        with pytest.raises(ValueError):
            UserInput(budget="lots of money")
