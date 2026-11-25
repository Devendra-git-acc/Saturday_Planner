"""Tool 6 - draftPlan.

Two interchangeable drafters share one contract (a DraftPlan of candidate ids in visiting order):
  * llm_draft  - asks the model to choose and order stops from the vetted candidate list
  * rule_plan  - deterministic beam search over the same candidates, using the same scheduler

The rule planner is the safety net (no key, outage, bad JSON, exhausted repairs) and also a
yardstick: if the LLM's plan scores worse than the rule plan, the rule plan wins.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import geo
from ..llm import LLM, call_validated
from ..prompts import draft_messages
from ..rules import MEAL_WINDOWS, meal_ok
from ..schemas import Candidate, CityData, CostReport, DraftPlan, DraftStop, LLMDraft, Place, Preferences
from ..trace import ToolInfo
from ..util import overlap_minutes
from .costing import estimate_cost, schedule_stop
from .options import TOP_FOR_PROMPT, interest_points
from .preferences import set_window
from .validate import validate_plan

START_CHOICES = (10 * 60, 12 * 60, 16 * 60)
MAX_HOP_MIN = 45
STRETCH_STEP = 15  # minutes added to one stop at a time when there is spare time
STRETCH_TARGET = 0.92  # stop lingering once this share of the window is planned


@dataclass
class _S:
    t: int
    prev: Place | None
    used: frozenset
    seq: tuple  # ((place, duration, start, end), ...)
    cost: int
    value: float
    tags: frozenset
    food_n: int


def _bucket(minute: int) -> str | None:
    for m, (a, b) in MEAL_WINDOWS.items():
        if a <= minute < b:
            return m
    return None


def _required_meals(prefs: Preferences, cands: list[Candidate]) -> list[str]:
    if not any(c.place.kind == "food" for c in cands):
        return []
    return [
        m for m in ("lunch", "dinner")
        if overlap_minutes(prefs.start_min or 0, prefs.end_min or 0, *MEAL_WINDOWS[m]) >= 90
    ]


def _terminal(s: _S, prefs: Preferences, required: list[str]) -> float:
    used = s.seq[-1][3] - (prefs.start_min or 0)
    val = s.value + 4.0 * min(1.0, used / max(1, prefs.minutes or 240))
    for m in required:
        ws, we = MEAL_WINDOWS[m]
        has = any(p.kind == "food" and a < we and b > ws for p, _, a, b in s.seq)
        val += 3.0 if has else -4.0
    n = len(s.seq)
    if n > 4:
        val -= 0.5 * (n - 4)
    if n < 2 and (prefs.minutes or 0) >= 150:
        val -= 2.0
    return val


def _search(prefs: Preferences, city: CityData, cands: list[Candidate], beam: int = 36) -> tuple[_S | None, float]:
    start, end = prefs.start_min, prefs.end_min
    if start is None or end is None:
        return None, float("-inf")
    cap = 5 if prefs.energy == "low" else 6
    budget = prefs.budget_inr
    required = _required_meals(prefs, cands)
    frontier = [_S(start, None, frozenset(), (), 0, 0.0, frozenset(), 0)]
    best: _S | None = None
    best_val = float("-inf")

    for _ in range(cap):
        nxt: dict[frozenset, _S] = {}
        for st in frontier:
            for c in cands:
                p = c.place
                if p.id in st.used or (p.kind == "food" and st.food_n >= 3):
                    continue
                sc = schedule_stop(city, prefs, st.prev, st.t, p, None, end)
                if sc.closed or sc.end > end or sc.travel.minutes > MAX_HOP_MIN:
                    continue
                if prefs.avoid_crowds and sc.crowd_level == "high":
                    continue
                cost = st.cost + sc.cost + sc.travel.cost
                if budget is not None and cost > budget:
                    continue
                v = c.score - 0.05 * sc.travel.minutes
                if p.kind == "food":
                    v += 1.0 if meal_ok(p, sc.start, sc.end) else -3.0
                    if sc.duration >= 35:
                        b = _bucket((sc.start + sc.end) // 2)
                        if b is not None and any(
                            q.kind == "food" and d >= 35 and _bucket((a + e) // 2) == b for q, d, a, e in st.seq
                        ):
                            v -= 3.0  # a second sit-down meal in the same meal window
                    if st.prev is not None and st.prev.kind == "food" and sc.duration >= 30 and st.seq[-1][1] >= 30:
                        v -= 2.5
                elif st.prev is not None and st.prev.category == p.category:
                    v -= 1.0
                # interests: big credit the first time one is covered, little for stacking the same one
                v -= interest_points(p, prefs)
                hits = [t for t in p.tags if t in prefs.interests]
                v += sum(6.0 if t not in st.tags else 0.5 for t in hits)
                if not hits and p.kind == "activity":
                    v -= 1.0  # a stop that matches nothing the person asked for has to earn its place
                ns = _S(
                    sc.end, p, st.used | {p.id}, st.seq + ((p, sc.duration, sc.start, sc.end),), cost,
                    st.value + v, st.tags | frozenset(p.tags), st.food_n + (1 if p.kind == "food" else 0),
                )
                old = nxt.get(ns.used)
                if old is None or ns.value > old.value:
                    nxt[ns.used] = ns
        if not nxt:
            break
        frontier = sorted(nxt.values(), key=lambda s: -s.value)[:beam]
        for s in frontier:
            tv = _terminal(s, prefs, required)
            if tv > best_val:
                best, best_val = s, tv
    return best, best_val


def stretch_to_window(prefs: Preferences, city: CityData, plan: DraftPlan, *, food_available: bool = True) -> DraftPlan:
    """Use spare time by letting the person linger, rather than padding the plan with stops they did not ask for.

    Adds STRETCH_STEP minutes at a time to whichever stop is furthest from its own maximum stay, and only
    keeps a change if the whole plan is still valid (open, within budget and window, not crowded) and
    scores better. Never goes past a venue's maximum, and stops at STRETCH_TARGET of the window.
    """
    if prefs.start_min is None or prefs.end_min is None or not prefs.minutes or not plan.stops:
        return plan
    by_id = city.by_id()

    def judge(durations: list[tuple[str, int]]):
        rep = estimate_cost(city, prefs, [DraftStop(place_id=i, duration_min=d) for i, d in durations])
        if len(rep.stops) != len(durations):
            return None
        v = validate_plan(rep, prefs, city, food_available=food_available)
        return (rep, v) if v.ok else None

    first = judge([(st.place_id, st.duration_min or by_id[st.place_id].duration) for st in plan.stops])
    if first is None:
        return plan
    rep, val = first
    target_end = prefs.start_min + int(STRETCH_TARGET * prefs.minutes)
    for _ in range(48):
        if rep.end >= target_end:
            break
        options = []
        for i, st in enumerate(rep.stops):
            venue = by_id[st.id]
            if st.duration + STRETCH_STEP > venue.max_dur:
                continue
            trial = [(x.id, x.duration + (STRETCH_STEP if j == i else 0)) for j, x in enumerate(rep.stops)]
            res = judge(trial)
            if res is not None and res[1].penalty < val.penalty:
                options.append((res[1].penalty, st.duration / venue.max_dur, i, res))
        if not options:
            break
        options.sort(key=lambda t: t[:3])
        rep, val = options[0][3]
    return DraftPlan(
        stops=[DraftStop(place_id=st.id, duration_min=st.duration, reason="") for st in rep.stops],
        notes=plan.notes,
    )


def rule_plan(prefs: Preferences, city: CityData, acts: list[Candidate], foods: list[Candidate]) -> DraftPlan | None:
    best, _ = _search(prefs, city, [*acts, *foods])
    if best is None:
        return None
    plan = DraftPlan(
        stops=[DraftStop(place_id=p.id, duration_min=d, reason="") for p, d, _, _ in best.seq],
        notes="deterministic planner",
    )
    return stretch_to_window(prefs, city, plan, food_available=bool(foods))


def choose_start(prefs: Preferences, city: CityData, acts: list[Candidate], foods: list[Candidate]) -> tuple[int, dict[int, float]]:
    """No start time given: try a few and keep the one where the best plan scores highest."""
    results: dict[int, float] = {}
    for s in START_CHOICES:
        if s + 60 > 23 * 60:
            continue
        pp = prefs.model_copy(deep=True)
        set_window(pp, s)
        st, val = _search(pp, city, [*acts, *foods])
        if st is not None:
            results[s] = round(val, 2)
    if not results:
        return START_CHOICES[0], {}
    best = max(results, key=lambda k: (results[k], -k))
    return best, results


def llm_draft(
    llm: LLM,
    prefs: Preferences,
    city: CityData,
    acts: list[Candidate],
    foods: list[Candidate],
    info: ToolInfo,
    *,
    previous: CostReport | None = None,
    problems: list[str] | None = None,
) -> DraftPlan:
    a = acts[: TOP_FOR_PROMPT["activity"]]
    f = foods[: TOP_FOR_PROMPT["food"]]
    matrix = geo.area_matrix(city, [c.place.area for c in (*a, *f)])
    msgs = draft_messages(prefs, city, a, f, matrix, previous=previous, problems=problems)
    res = call_validated(llm, "draft", msgs, LLMDraft, effort="low", info=info)
    return DraftPlan(
        stops=[DraftStop(place_id=s.id, duration_min=s.duration_min, reason=(s.reason or "")[:160]) for s in res.stops[:8]],
        notes=(res.notes or "")[:200],
    )
