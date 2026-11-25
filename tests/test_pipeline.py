"""The deterministic tools, one at a time: parse -> options -> estimateCost -> validatePlan -> rule planner."""
from __future__ import annotations

import pytest

from app.cities import get_city
from app.llm import LLMBadOutput
from app.schemas import DraftStop, LLMPrefs, Preferences, UserInput
from app.tools.costing import MAX_WAIT, estimate_cost
from app.tools.options import get_activity_options, get_food_options, summarize
from app.tools.finalize import trade_off_facts
from app.tools.planner import START_CHOICES, choose_start, llm_draft, rule_plan, stretch_to_window
from app.tools.preferences import (
    DEFAULT_BUDGET, DEFAULT_MINUTES, finalize_defaults, merge_llm, parse_user_preferences, rules_parse, set_window,
)
from app.tools.validate import user_cautions, validate_plan
from app.trace import ToolInfo
from app.util import parse_clock_range

from .conftest import ASSIGNMENT, ScriptedLLM, draft

BLR = get_city("bangalore")


def mk(start: int | None = 10 * 60, minutes: int = 240, **kw) -> Preferences:
    kw.setdefault("budget_inr", 2000)
    kw.setdefault("interests", ["food", "walks"])
    p = Preferences(city="bangalore", city_display="Bangalore", city_known=True, minutes=minutes, **kw)
    if start is not None:
        set_window(p, start)
    return p


def ds(*ids: str) -> list[DraftStop]:
    return [DraftStop(place_id=i) for i in ids]


def judge(prefs: Preferences, *ids: str):
    report = estimate_cost(BLR, prefs, ds(*ids))
    return report, validate_plan(report, prefs, BLR)


def codes(v, severity=None) -> set[str]:
    return {i.code for i in v.issues if severity is None or i.severity == severity}


# =============================================================== parseUserPreferences


def parse(**kw) -> Preferences:
    return rules_parse(UserInput(**kw))


def test_assignment_example_is_understood_without_a_model():
    p = parse(**ASSIGNMENT)
    assert (p.city, p.budget_inr, p.minutes, p.energy) == ("bangalore", 2000, 240, "low")
    assert p.diet == "vegetarian" and p.avoid_crowds
    assert p.interests == ["food", "music", "walks"]
    assert p.parsed_by == "rules" and not p.other_constraints


@pytest.mark.parametrize("text,diet", [
    ("vegan please", "vegan"), ("I am Jain", "jain"), ("pure veg only", "vegetarian"), ("no meat", "vegetarian"),
    ("eggetarian", "vegetarian"), ("non-vegetarian is fine", "none"), ("non veg", "none"), ("anything goes", "none"),
])
def test_diet_words(text, diet):
    assert parse(city="Pune", free_text=text).diet == diet


@pytest.mark.parametrize("text,flag", [
    ("avoid crowds", "avoid_crowds"), ("no crowds please", "avoid_crowds"), ("crowd-free", "avoid_crowds"),
    ("no alcohol", "avoid_alcohol"), ("teetotal", "avoid_alcohol"), ("no pubs", "avoid_alcohol"),
    ("indoors only", "indoor_only"), ("it is too hot outside", "indoor_only"), ("avoid the sun", "indoor_only"),
])
def test_constraint_words(text, flag):
    assert getattr(parse(city="Pune", constraints=[text]), flag) is True


@pytest.mark.parametrize("text,energy,social", [
    ("tired but wants to do something fun", "low", "balanced"), ("exhausted and sleepy", "low", "balanced"),
    ("energetic and adventurous", "high", "balanced"), ("lively and social", "medium", "lively"),
    ("quiet and peaceful", "medium", "quiet"), ("tired and energetic", "medium", "balanced"),  # contradictory: no guess
])
def test_mood_words(text, energy, social):
    p = parse(city="Pune", mood=text)
    assert (p.energy, p.social) == (energy, social)


def test_unsupported_constraints_are_kept_visible_instead_of_dropped():
    p = parse(city="Pune", constraints=["wheelchair accessible", "vegetarian", "no pubs"])
    assert p.other_constraints == ["wheelchair accessible"]
    assert p.diet == "vegetarian" and p.avoid_alcohol


def test_unknown_interests_are_reported_not_silently_lost():
    p = parse(city="Pune", interests=["food", "underwater basket weaving"])
    assert p.interests == ["food"]
    assert any("underwater basket weaving" in n for n in p.notes)


def test_free_text_carries_everything_when_no_form_is_filled():
    p = parse(free_text="I'm in Pune, vegan, 3 hours, around Rs 1,500, starting at 4pm, love books and coffee, no alcohol, indoors")
    assert (p.city, p.diet, p.minutes, p.budget_inr, p.start_min) == ("pune", "vegan", 180, 1500, 16 * 60)
    assert p.interests == ["coffee", "books"] and p.avoid_alcohol and p.indoor_only


@pytest.mark.parametrize("free,start,minutes", [
    ("free 4-7pm", 16 * 60, 180), ("from 10am to 2pm", 10 * 60, 240), ("I can go 11am-1pm", 11 * 60, 120),
    ("16:00-19:30", 16 * 60, 210),
])
def test_a_clock_window_sets_both_start_and_length(free, start, minutes):
    p = parse(city="Pune", free_text=free)
    assert (p.start_min, p.minutes) == (start, minutes)
    assert any("I read your window as" in a for a in p.assumptions)


def test_a_stated_duration_beats_the_length_implied_by_a_window():
    p = parse(city="Pune", available_time="2 hours", free_text="starting at 4pm")
    assert (p.start_min, p.minutes) == (16 * 60, 120)


def test_start_time_field_wins_over_a_conflicting_window_and_says_so():
    p = parse(city="Pune", start_time="11:00", free_text="free 4-7pm")
    assert p.start_min == 11 * 60 and p.minutes is None
    assert any("doesn't match" in a for a in p.assumptions)


@pytest.mark.parametrize("text", ["2-3 hours", "5-6 people", "Rs 1,500-2,000", "4-7"])
def test_ranges_that_are_not_clock_times_are_ignored(text):
    assert parse_clock_range(text) is None


def test_typos_and_ambiguity_in_the_city_field():
    typo = parse(city="Banglore")
    assert typo.city == "bangalore" and any("I read 'Banglore' as Bangalore" in a for a in typo.assumptions)
    both = parse(city="Pune or Bangalore")
    assert both.city is None and both.city_ambiguous and not both.city_known
    unknown = parse(city="Mumbai")
    assert unknown.city == "mumbai" and unknown.city_display == "Mumbai" and not unknown.city_known


def test_finalize_defaults_states_every_assumption():
    p = finalize_defaults(Preferences())
    assert (p.budget_inr, p.minutes, p.interests) == (DEFAULT_BUDGET, DEFAULT_MINUTES, ["food", "walks"])
    assert set(p.defaulted) == {"budget", "time", "interests"}
    assert len(p.assumptions) == 3


@pytest.mark.parametrize("budget,minutes,expect_budget,expect_minutes", [
    (5_000_000, 5, 1_000_000, 30), (-50, 5000, 0, 720), (0, 90, 0, 90),
])
def test_finalize_defaults_clamps_silly_values(budget, minutes, expect_budget, expect_minutes):
    p = finalize_defaults(Preferences(budget_inr=budget, minutes=minutes, interests=["food"]))
    assert (p.budget_inr, p.minutes) == (expect_budget, expect_minutes)


def test_set_window_never_runs_past_midnight_and_says_so():
    p = Preferences(minutes=300)
    set_window(p, 22 * 60)
    assert p.end_min == 23 * 60 + 59 and p.minutes == 119
    assert p.assumptions == ["That would run past midnight, so I planned until 11:59 pm."]
    set_window(p, 22 * 60)
    assert len(p.assumptions) == 1  # no repeats
    fits = Preferences(minutes=120)
    set_window(fits, 18 * 60)
    assert fits.end_min == 20 * 60 and not fits.assumptions


def test_a_very_late_start_is_pulled_back_and_disclosed():
    q = finalize_defaults(Preferences(minutes=120, interests=["food"], start_min=23 * 60 + 30, budget_inr=500))
    assert q.start_min == 22 * 60 and q.end_min == 23 * 60 + 59
    assert any("no later than 10:00 pm" in a for a in q.assumptions)


class TestMergeKeepsHardConstraints:
    """The model reads mood better than keyword rules do, but may only ever ADD constraints."""

    def base(self, **kw):
        return rules_parse(UserInput(city="Pune", constraints=["vegetarian", "avoid crowds"], **kw))

    def test_model_cannot_relax_what_the_rules_found(self):
        merged = merge_llm(self.base(), LLMPrefs(diet="none", avoid_crowds=False, avoid_alcohol=False), UserInput())
        assert merged.diet == "vegetarian" and merged.avoid_crowds

    def test_model_can_tighten_but_not_loosen_diet(self):
        assert merge_llm(self.base(), LLMPrefs(diet="vegan"), UserInput()).diet == "vegan"
        assert merge_llm(self.base(), LLMPrefs(diet="vegetarian"), UserInput()).diet == "vegetarian"
        assert merge_llm(self.base(), LLMPrefs(diet="jain"), UserInput()).diet == "jain"

    def test_model_may_add_alcohol_and_indoor_limits(self):
        merged = merge_llm(self.base(), LLMPrefs(avoid_alcohol=True, indoor_only=True), UserInput())
        assert merged.avoid_alcohol and merged.indoor_only

    def test_model_decides_mood_but_not_the_city_the_user_typed(self):
        merged = merge_llm(self.base(), LLMPrefs(city="Mumbai", energy="high", social="lively"), UserInput())
        assert merged.city == "pune" and (merged.energy, merged.social) == ("high", "lively")

    def test_model_cannot_resolve_an_ambiguous_city(self):
        amb = rules_parse(UserInput(city="Pune or Bangalore"))
        assert merge_llm(amb, LLMPrefs(city="Bangalore"), UserInput()).city is None

    def test_model_extras_are_deduped_trimmed_and_capped(self):
        base = self.base()
        base.other_constraints = ["step-free access"]
        extras = ["Step-free access", "x" * 200, "no stairs", "kid friendly", "pet friendly", "near metro", "vegan only"]
        merged = merge_llm(base, LLMPrefs(other_constraints=extras), UserInput())
        assert len(merged.other_constraints) == 5
        assert all(len(c) <= 80 for c in merged.other_constraints)
        assert [c.lower() for c in merged.other_constraints].count("step-free access") == 1
        assert not any("vegan" in c.lower() for c in merged.other_constraints)  # diet is handled as a diet, not a free-form note

    def test_model_numbers_only_fill_gaps(self):
        base = parse(city="Pune", budget=900)
        merged = merge_llm(base, LLMPrefs(budget_inr=5000, minutes=200, start_time="5 pm"), UserInput())
        assert merged.budget_inr == 900 and merged.minutes == 200 and merged.start_min == 17 * 60


def test_parse_tool_skips_the_model_when_structured_input_is_clear():
    llm = ScriptedLLM(parse={"diet": "vegan"})
    info = ToolInfo()
    p = parse_user_preferences(UserInput(**{**ASSIGNMENT, "mood": None}), llm, info)
    assert llm.count("parse") == 0 and p.parsed_by == "rules"
    assert any("no model call needed" in n for n in info.notes)


def test_parse_tool_uses_the_model_for_messy_text_and_merges():
    llm = ScriptedLLM(parse={"energy": "low", "social": "quiet", "diet": "vegan", "interests": ["books"]})
    p = parse_user_preferences(UserInput(city="Pune", free_text="I'm wiped out, somewhere calm"), llm, ToolInfo())
    assert llm.count("parse") == 1 and p.parsed_by == "llm"
    assert (p.energy, p.social, p.diet) == ("low", "quiet", "vegan") and "books" in p.interests


def test_parse_tool_falls_back_to_rules_when_the_model_is_unusable():
    llm = ScriptedLLM(parse=LLMBadOutput("nope"))
    info = ToolInfo()
    p = parse_user_preferences(UserInput(city="Pune", free_text="tired and want something quiet"), llm, info)
    assert p.parsed_by == "rules" and p.energy == "low" and p.social == "quiet"
    assert info.fallback and "keyword rules" in info.fallback


# =============================================================== getActivityOptions / getFoodOptions


def kept(res) -> set[str]:
    return {c.place.id for c in res.candidates}


def test_options_never_mix_kinds_and_are_ranked_best_first():
    p = mk(interests=["music"])
    acts, foods = get_activity_options(p, BLR), get_food_options(p, BLR)
    assert all(c.place.kind == "activity" for c in acts.candidates) and all(c.place.kind == "food" for c in foods.candidates)
    scores = [c.score for c in acts.candidates]
    assert scores == sorted(scores, reverse=True)
    assert acts.candidates[0].place.id == "blr-ime"  # the only music venue open at 10 am
    assert any("music" in r for r in acts.candidates[0].reasons)


@pytest.mark.parametrize("prefs,res_fn,place_id,reason", [
    (dict(diet="vegetarian"), get_food_options, "blr-nagarjuna", "suit a vegetarian diet"),
    (dict(diet="vegan"), get_food_options, "blr-koshys", "suit a vegan diet"),  # veg-friendly, but not vegan-adaptable
    (dict(avoid_alcohol=True, start=13 * 60), get_food_options, "blr-toit", "built around alcohol"),
    (dict(avoid_alcohol=True, start=13 * 60), get_activity_options, "blr-windmills", "built around alcohol"),
    (dict(indoor_only=True), get_activity_options, "blr-cubbon", "outdoors"),
    (dict(budget_inr=200), get_activity_options, "blr-palace", "over your budget"),
    (dict(budget_inr=200), get_food_options, "blr-koshys", "over your budget"),
    (dict(start=6 * 60, minutes=120), get_activity_options, "blr-palace", "closed during your window"),
    (dict(start=10 * 60, minutes=40), get_activity_options, "blr-lalbagh", "too long for your time"),
    (dict(avoid_crowds=True), get_activity_options, "blr-commercial", "crowded on Saturdays"),
])
def test_hard_constraints_remove_venues_and_count_the_reason(prefs, res_fn, place_id, reason):
    start = prefs.pop("start", 10 * 60)
    p = mk(start=start, **prefs)
    res = res_fn(p, BLR)
    assert place_id not in kept(res)
    assert any(reason in k for k in res.dropped), res.dropped
    assert len(res.candidates) + sum(res.dropped.values()) == res.pool_size


def test_a_live_music_venue_is_only_offered_when_the_music_is_on():
    """Windmills is a restaurant at noon, but the reason to go is the live jazz, which starts in the evening."""
    day = get_activity_options(mk(start=12 * 60, minutes=240, budget_inr=3000, interests=["music", "nightlife"]), BLR)
    assert "blr-windmills" not in kept(day) and day.dropped.get("no live music during your window") == 1
    night = get_activity_options(mk(start=19 * 60, minutes=180, budget_inr=3000, interests=["music", "nightlife"]), BLR)
    assert "blr-windmills" in kept(night)


def test_the_midday_music_lover_is_never_sent_to_a_live_music_bar():
    for city, bar in (("bangalore", "blr-windmills"), ("pune", "pun-highspirits")):
        c = get_city(city)
        for start in (9 * 60, 11 * 60, 13 * 60, 15 * 60):
            p = Preferences(city=city, city_display=city.title(), city_known=True, minutes=240, budget_inr=3000, interests=["music", "food", "nightlife"])
            set_window(p, start)
            acts, foods = get_activity_options(p, c), get_food_options(p, c)
            plan = rule_plan(p, c, acts.candidates, foods.candidates)
            assert plan is not None and bar not in {s.place_id for s in plan.stops}, (city, start)


def test_the_live_music_reason_reads_naturally_in_the_trade_offs():
    p = mk(start=12 * 60, minutes=240, budget_inr=3000, interests=["music", "nightlife"])
    skipped = [{"name": "Windmills Craftworks (live jazz)", "reason": "no live music during your window"}]
    facts = trade_off_facts(p, estimate_cost(BLR, p, ds("blr-ime")), skipped, [], [])
    assert any("Windmills Craftworks (live jazz) because it has no live music during your time window" in f for f in facts), facts


def test_live_only_venues_start_in_the_evening_in_the_data():
    flagged = [pl for key in ("bangalore", "pune", "gurugram") for pl in get_city(key).places if pl.live_only]
    assert {pl.id for pl in flagged} == {"blr-windmills", "pun-highspirits"}
    from app.util import parse_windows

    for pl in flagged:
        assert all(start >= 17 * 60 for start, _ in parse_windows(pl.hours)), pl.id


def test_hard_filters_never_let_a_violating_venue_through():
    p = mk(diet="vegetarian", avoid_alcohol=True, indoor_only=True, avoid_crowds=True, budget_inr=500, start=11 * 60)
    for res in (get_activity_options(p, BLR), get_food_options(p, BLR)):
        for c in res.candidates:
            pl = c.place
            assert pl.cost <= 500 and not pl.alcohol_centric and pl.diet != "non_veg_focus"
            assert pl.indoor or pl.kind == "food"


def test_soft_preferences_change_the_order_not_the_membership():
    quiet = get_activity_options(mk(social="quiet"), BLR)
    lively = get_activity_options(mk(social="lively"), BLR)
    assert kept(quiet) == kept(lively)
    ids = [c.place.id for c in quiet.candidates]
    assert ids.index("blr-commercial") > ids.index("blr-cubbon")


def test_cautions_travel_with_candidates_that_pass_with_a_caveat():
    p = mk(diet="vegan")
    foods = get_food_options(p, BLR)
    flagged = [c for c in foods.candidates if c.flags]
    assert flagged and all("vegan" in f.lower() for c in flagged for f in c.flags)


def test_notable_removals_name_venues_the_user_would_have_liked():
    res = get_activity_options(mk(interests=["music"], indoor_only=True, budget_inr=100), BLR)
    assert {"name": "Indian Music Experience", "reason": "over your budget"} in res.skipped_notable


def test_summarize_states_what_was_removed():
    res = get_food_options(mk(diet="vegetarian", budget_inr=300), BLR)
    text = summarize(res)
    assert text.startswith(f"{len(res.candidates)} of {res.pool_size} venues kept") and "removed" in text
    everything = get_activity_options(Preferences(budget_inr=1_000_000, minutes=600, interests=["food"]), BLR)
    assert summarize(everything) == "16 of 16 venues kept"  # nothing removed: no clutter


# =============================================================== estimateCost


def test_a_first_stop_has_no_travel_and_the_totals_add_up():
    r, _ = judge(mk(), "blr-cubbon", "blr-tipu", "blr-adigas")
    assert r.stops[0].travel.mode == "start" and r.stops[0].travel.minutes == 0
    assert r.total_cost == r.spend + r.travel_cost == sum(s.cost + s.travel.cost for s in r.stops)
    assert r.end == r.stops[-1].end and r.total_minutes == r.end - r.start
    assert r.travel_minutes == sum(s.travel.minutes for s in r.stops)
    for a, b in zip(r.stops, r.stops[1:]):
        assert b.start >= a.end + b.travel.minutes  # travel (and any wait) is real time


def test_unknown_and_repeated_ids_become_errors_instead_of_exceptions():
    r, v = judge(mk(), "made-up", "blr-cubbon", "blr-cubbon")
    assert [s.id for s in r.stops] == ["blr-cubbon"]
    assert {i.code for i in r.pre_issues} == {"unknown_stop", "duplicate_stop"}
    assert {"unknown_stop", "duplicate_stop"} <= codes(v, "error") and not v.ok


def test_waiting_for_a_venue_to_open_is_allowed_only_briefly():
    r, _ = judge(mk(start=10 * 60 + 10), "blr-blossoms")  # opens 10:30
    s = r.stops[0]
    assert (s.wait, s.start, s.closed) == (20, 10 * 60 + 30, False)
    late, _ = judge(mk(start=10 * 60), "blr-ranga")  # opens 19:30
    assert late.stops[0].closed and late.stops[0].warnings[0].startswith("Closed: ")
    assert MAX_WAIT == 30


def test_a_venue_that_is_already_shut_is_closed():
    r, v = judge(mk(start=19 * 60, minutes=120), "blr-palace")
    assert r.stops[0].closed and "closed" in codes(v, "error")


def test_stays_are_trimmed_to_the_closing_time_and_the_window():
    r, _ = judge(mk(start=18 * 60, minutes=60), "blr-cubbon")  # closes 18:30
    assert (r.stops[0].duration, r.stops[0].end) == (30, 18 * 60 + 30)
    assert any(w.startswith("Shortened") for w in r.stops[0].warnings)
    r2, _ = judge(mk(start=10 * 60, minutes=60), "blr-cubbon")  # 75-minute park, 60-minute window
    assert r2.stops[0].end <= 11 * 60


def test_requested_durations_are_clamped_to_what_the_venue_allows():
    p = mk(minutes=480)
    short = estimate_cost(BLR, p, [DraftStop(place_id="blr-palace", duration_min=1)])
    long = estimate_cost(BLR, p, [DraftStop(place_id="blr-palace", duration_min=999)])
    pl = BLR.by_id()["blr-palace"]
    assert short.stops[0].duration == pl.min_dur and long.stops[0].duration == pl.max_dur


def test_far_apart_stops_cost_real_money_and_time_and_evenings_are_slower():
    day, _ = judge(mk(start=10 * 60), "blr-cubbon", "blr-ime")
    eve, _ = judge(mk(start=17 * 60, minutes=300), "blr-cubbon", "blr-ime")
    assert day.stops[1].travel.cost >= 60 and day.stops[1].travel.mode == "auto/cab"
    assert eve.stops[1].travel.minutes > day.stops[1].travel.minutes


def test_estimate_without_a_start_assumes_ten_am():
    p = mk(start=None)
    assert estimate_cost(BLR, p, ds("blr-cubbon")).start == 10 * 60


# =============================================================== validatePlan


@pytest.mark.parametrize("prefs,start,ids,expected", [
    (dict(diet="vegetarian"), 11 * 60, ("blr-nagarjuna",), "diet"),
    (dict(avoid_alcohol=True), 13 * 60, ("blr-toit",), "alcohol"),
    (dict(indoor_only=True), 10 * 60, ("blr-cubbon",), "indoor"),
    (dict(avoid_crowds=True), 10 * 60, ("blr-commercial",), "crowded"),
    (dict(budget_inr=100), 10 * 60, ("blr-palace",), "over_budget"),
    (dict(minutes=40), 10 * 60, ("blr-lalbagh",), "over_time"),
    (dict(), 10 * 60, ("blr-ranga",), "closed"),
    (dict(), 10 * 60, (), "empty"),
    (dict(budget_inr=5000, minutes=300), 17 * 60, ("blr-ime", "blr-windmills"), "long_hop"),
])
def test_every_hard_constraint_is_an_error(prefs, start, ids, expected):
    _, v = judge(mk(start=start, **prefs), *ids)
    assert expected in codes(v, "error") and not v.ok and not v.clean
    assert v.penalty >= 100


def test_a_valid_plan_is_ok_and_a_good_one_is_clean():
    _, v = judge(mk(diet="vegetarian", energy="low"), "blr-cubbon", "blr-adigas", "blr-tipu")
    assert v.ok and not v.errors
    good = judge(mk(start=10 * 60, minutes=210, interests=["walks", "food"]), "blr-cubbon", "blr-adigas", "blr-tipu")[1]
    assert good.ok and good.clean and good.penalty < 20


def test_quality_problems_are_major_warnings_but_not_errors():
    quiet = judge(mk(social="quiet"), "blr-commercial")[1]
    assert quiet.ok and any(i.code == "noise" and i.major for i in quiet.issues)
    gap = judge(mk(start=10 * 60, minutes=300), "blr-cubbon", "blr-tipu")[1]
    assert gap.ok and any(i.code == "meal_gap" and i.major for i in gap.issues)
    one = judge(mk(), "blr-palace")[1]
    assert {"too_few", "underfilled"} <= codes(one, "warn") and not one.clean


def test_minor_notes_do_not_block_a_clean_verdict():
    tired = judge(mk(energy="low", start=10 * 60, minutes=180, interests=["shopping"]), "blr-commercial", "blr-ich")[1]
    energy = [i for i in tired.issues if i.code == "energy"]
    assert energy and not energy[0].major  # a one-step energy gap is only worth a mention


def test_back_to_back_sit_down_meals_are_flagged():
    _, v = judge(mk(start=11 * 60), "blr-ich", "blr-glens")
    assert "back_to_back_food" in codes(v, "warn")


def test_too_many_stops_is_major_and_the_cap_is_lower_when_tired():
    ids = ("blr-cubbon", "blr-tipu", "blr-lalbagh", "blr-bull", "blr-blossoms", "blr-palace")
    assert "too_many" not in codes(judge(mk(start=9 * 60, minutes=600), *ids)[1])
    assert "too_many" in codes(judge(mk(start=9 * 60, minutes=600, energy="low"), *ids)[1])


def test_interest_coverage_is_info_when_partial_and_a_warning_when_total():
    partial = judge(mk(interests=["walks", "music"], start=10 * 60, minutes=120), "blr-cubbon", "blr-tipu")[1]
    assert any(i.code == "interests" and i.severity == "info" for i in partial.issues)
    none = judge(mk(interests=["music"], start=10 * 60, minutes=120), "blr-cubbon", "blr-tipu")[1]
    assert any(i.code == "interests" and i.severity == "warn" and i.major for i in none.issues)


def test_meal_gap_is_skipped_when_there_is_no_food_to_offer():
    p = mk(start=10 * 60, minutes=300)
    report = estimate_cost(BLR, p, ds("blr-cubbon", "blr-tipu"))
    assert "meal_gap" in codes(validate_plan(report, p, BLR))
    assert "meal_gap" not in codes(validate_plan(report, p, BLR, food_available=False))


def test_a_better_plan_always_has_a_lower_penalty_than_a_broken_one():
    p = mk(diet="vegetarian", start=10 * 60, minutes=210)
    good = judge(p, "blr-cubbon", "blr-adigas", "blr-tipu")[1]
    bad = judge(p, "blr-nagarjuna")[1]
    assert good.penalty < bad.penalty


class TestUserCautions:
    def test_reader_friendly_wording_replaces_planner_instructions(self):
        _, v = judge(mk(), "blr-palace")
        out = user_cautions(v)
        assert "Only one stop fit within your limits, so much of your time window is free." in out
        assert not any("add something" in c or "add another" in c for c in out)

    def test_underfilled_is_hidden_when_too_few_already_says_it(self):
        _, v = judge(mk(), "blr-palace")
        assert {"too_few", "underfilled"} <= codes(v)
        assert not any("spare time" in c for c in user_cautions(v))

    def test_underfilled_shows_when_it_stands_alone(self):
        _, v = judge(mk(start=11 * 60, minutes=300), "blr-ich", "blr-glens")
        assert any("spare time" in c for c in user_cautions(v))

    def test_only_warnings_are_shown_and_never_more_than_the_limit(self):
        _, v = judge(mk(budget_inr=100, social="quiet", energy="low"), "blr-commercial")
        out = user_cautions(v, limit=2)
        assert len(out) == 2 and len(set(out)) == len(out)
        assert not any("budget" in c for c in user_cautions(v))  # that is an error: the plan would not be shown


# =============================================================== draftPlan


def assignment_prefs(start=10 * 60) -> Preferences:
    return finalize_defaults(rules_parse(UserInput(**ASSIGNMENT, start_time="10:00"))) if start == 10 * 60 else mk(start)


def test_rule_plan_satisfies_every_hard_constraint_for_the_assignment_example():
    p = assignment_prefs()
    acts, foods = get_activity_options(p, BLR), get_food_options(p, BLR)
    plan = rule_plan(p, BLR, acts.candidates, foods.candidates)
    assert plan and 2 <= len(plan.stops) <= 5
    report = estimate_cost(BLR, p, plan.stops)
    v = validate_plan(report, p, BLR)
    assert v.ok, [i.message for i in v.errors]
    assert report.total_cost <= p.budget_inr and report.end <= p.end_min
    assert any(s.kind == "food" for s in report.stops)  # a 10 am to 2 pm window covers lunch
    assert all(BLR.by_id()[s.id].diet != "non_veg_focus" for s in report.stops)


def test_rule_plan_is_deterministic():
    p = assignment_prefs()
    acts, foods = get_activity_options(p, BLR), get_food_options(p, BLR)
    first = rule_plan(p, BLR, acts.candidates, foods.candidates)
    second = rule_plan(p, BLR, acts.candidates, foods.candidates)
    assert [s.place_id for s in first.stops] == [s.place_id for s in second.stops]


def test_rule_plan_returns_none_when_nothing_is_open():
    p = mk(start=2 * 60, minutes=60)  # 2 am: every venue is shut
    acts, foods = get_activity_options(p, BLR), get_food_options(p, BLR)
    assert rule_plan(p, BLR, acts.candidates, foods.candidates) is None


def test_rule_plan_needs_a_window():
    p = mk(start=None)
    acts, foods = get_activity_options(p, BLR), get_food_options(p, BLR)
    assert rule_plan(p, BLR, acts.candidates, foods.candidates) is None


def test_rule_plan_covers_a_second_interest_before_repeating_the_first():
    p = mk(interests=["walks", "music"], start=10 * 60, minutes=300, budget_inr=3000)
    acts, foods = get_activity_options(p, BLR), get_food_options(p, BLR)
    ids = [s.place_id for s in rule_plan(p, BLR, acts.candidates, foods.candidates).stops]
    tags = {t for i in ids for t in BLR.by_id()[i].tags}
    assert {"walks", "music"} <= tags


def test_choose_start_compares_the_fixed_options_and_picks_the_best_score():
    p = mk(start=None, interests=["music", "food"], diet="vegetarian")
    acts, foods = get_activity_options(p, BLR), get_food_options(p, BLR)
    start, scores = choose_start(p, BLR, acts.candidates, foods.candidates)
    assert start in START_CHOICES and set(scores) <= set(START_CHOICES) and scores
    assert scores[start] == max(scores.values())
    assert p.start_min is None  # the comparison must not leak into the caller's preferences


def test_choose_start_has_a_default_when_nothing_works():
    p = mk(start=None, budget_inr=0, diet="vegan", indoor_only=True, minutes=30)
    acts, foods = get_activity_options(p, BLR), get_food_options(p, BLR)
    assert not acts.candidates and not foods.candidates
    assert choose_start(p, BLR, acts.candidates, foods.candidates) == (START_CHOICES[0], {})


def test_llm_draft_trims_what_the_model_sends_back():
    long_reason = "r" * 500
    reply = {"stops": [{"id": f"blr-cubbon", "duration_min": 60, "reason": long_reason}] + [{"id": f"x{i}"} for i in range(12)], "notes": "n" * 500}
    llm = ScriptedLLM(draft=reply)
    p = mk()
    acts, foods = get_activity_options(p, BLR), get_food_options(p, BLR)
    out = llm_draft(llm, p, BLR, acts.candidates, foods.candidates, ToolInfo())
    assert len(out.stops) == 8 and len(out.stops[0].reason) == 160 and len(out.notes) == 200


def test_llm_draft_only_shows_the_model_venues_that_passed_the_filters():
    llm = ScriptedLLM(draft=draft("blr-cubbon"))
    p = mk(diet="vegetarian")
    acts, foods = get_activity_options(p, BLR), get_food_options(p, BLR)
    llm_draft(llm, p, BLR, acts.candidates, foods.candidates, ToolInfo())
    prompt = "\n".join(m["content"] for m in llm.calls[0][1])
    assert "blr-cubbon" in prompt and "blr-nagarjuna" not in prompt and "Nagarjuna" not in prompt


def test_llm_draft_retries_once_with_the_error_then_raises():
    llm = ScriptedLLM(draft={"stops": []})
    p = mk()
    acts, foods = get_activity_options(p, BLR), get_food_options(p, BLR)
    info = ToolInfo()
    with pytest.raises(LLMBadOutput):
        llm_draft(llm, p, BLR, acts.candidates, foods.candidates, info)
    assert llm.count("draft") == 2 and info.retries == 1
    assert "could not be used" in llm.calls[1][1][-1]["content"]


# =============================================================== lingering instead of padding


def _plan(prefs, ids_and_minutes):
    from app.schemas import DraftPlan

    return DraftPlan(stops=[DraftStop(place_id=i, duration_min=m) for i, m in ids_and_minutes])


def test_spare_time_becomes_longer_stays_not_extra_stops():
    p = mk(start=10 * 60, minutes=360, diet="vegetarian", energy="low", interests=["walks", "history"])
    plan = _plan(p, [("blr-cubbon", 60), ("blr-adigas", 45), ("blr-tipu", 45)])
    out = stretch_to_window(p, BLR, plan)
    assert [s.place_id for s in out.stops] == [s.place_id for s in plan.stops]  # same stops, same order
    report = estimate_cost(BLR, p, out.stops)
    assert report.total_minutes > 0.8 * 360
    assert validate_plan(report, p, BLR).ok
    for s in report.stops:
        assert s.duration <= BLR.by_id()[s.id].max_dur  # nobody is asked to stay longer than a venue allows


def test_stretching_never_runs_past_closing_the_window_or_the_budget():
    p = mk(start=16 * 60, minutes=300, budget_inr=450)
    plan = _plan(p, [("blr-cubbon", 30), ("blr-adigas", 45)])
    report = estimate_cost(BLR, p, stretch_to_window(p, BLR, plan).stops)
    assert not any(s.closed for s in report.stops)
    assert report.end <= p.end_min and report.total_cost <= p.budget_inr
    cubbon = report.stops[0]
    assert cubbon.end <= 18 * 60 + 30  # the park closes at 6:30 pm


def test_stretching_respects_avoid_crowds():
    p = mk(start=10 * 60, minutes=300, avoid_crowds=True)
    out = stretch_to_window(p, BLR, _plan(p, [("blr-ime", 60), ("blr-ich", 45)]))
    report = estimate_cost(BLR, p, out.stops)
    assert all(s.crowd_level != "high" for s in report.stops) and validate_plan(report, p, BLR).ok


def test_a_short_window_is_left_alone_and_the_result_is_deterministic():
    p = mk(start=10 * 60, minutes=60)
    plan = _plan(p, [("blr-cubbon", 60)])
    assert [s.duration_min for s in stretch_to_window(p, BLR, plan).stops] == [60]
    long = mk(start=10 * 60, minutes=360)
    plan2 = _plan(long, [("blr-cubbon", 60), ("blr-tipu", 45)])
    a, b = stretch_to_window(long, BLR, plan2), stretch_to_window(long, BLR, plan2)
    assert [s.duration_min for s in a.stops] == [s.duration_min for s in b.stops]


def test_an_invalid_draft_is_returned_unchanged():
    p = mk(start=10 * 60, minutes=300, diet="vegetarian")
    bad = _plan(p, [("blr-nagarjuna", 60)])  # breaks the diet: stretching must not try to fix or hide it
    assert stretch_to_window(p, BLR, bad) is bad


def test_rule_plan_fills_most_of_a_long_open_day():
    p = mk(start=9 * 60, minutes=420, budget_inr=3000, interests=["walks", "history", "food"], energy="medium")
    acts, foods = get_activity_options(p, BLR), get_food_options(p, BLR)
    report = estimate_cost(BLR, p, rule_plan(p, BLR, acts.candidates, foods.candidates).stops)
    assert report.total_minutes >= 0.75 * 420 and validate_plan(report, p, BLR).ok


def test_the_summary_says_when_a_lot_of_the_window_is_left_open():
    p = mk(start=19 * 60 + 30, minutes=270, budget_inr=900)
    report = estimate_cost(BLR, p, ds("blr-ich"))
    facts = trade_off_facts(p, report, [], [], [])
    assert any("of your window is left open" in f for f in facts)
    busy = estimate_cost(BLR, mk(start=10 * 60, minutes=100), ds("blr-cubbon"))
    assert not any("left open" in f for f in trade_off_facts(mk(start=10 * 60, minutes=100), busy, [], [], []))
