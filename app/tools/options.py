"""Tools 2 and 3 - getActivityOptions / getFoodOptions.

Pure code, no LLM: hard constraints REMOVE a venue (with a counted reason), soft preferences only
change its score. The reasons we attach here are reused later to explain the plan, so the
explanation is always grounded in the same facts the filter used.
"""
from __future__ import annotations

from ..rules import (
    LEVELS,
    crowd_at,
    diet_check,
    energy_gap,
    noise_gap,
)
from ..schemas import Candidate, CityData, OptionsResult, Place, Preferences
from ..util import overlap_minutes, parse_windows

TOP_FOR_PROMPT = {"activity": 14, "food": 10}


def _open_overlap(p: Place, start: int, end: int) -> int:
    return max((overlap_minutes(s, e, start, end) for s, e in parse_windows(p.hours)), default=0)


def best_crowd(p: Place, start: int, end: int) -> str:
    """Calmest level the place reaches while it is open inside [start, end]."""
    best = 3
    t = start
    while t < end:
        if any(s <= t < e for s, e in parse_windows(p.hours)):
            best = min(best, LEVELS[crowd_at(p, t)])
        t += 30
    return ("low", "medium", "high")[best] if best < 3 else p.crowd


def interest_points(p: Place, prefs: Preferences) -> float:
    """Base credit for matching the person's interests (the planner re-weights this for variety)."""
    matched = [i for i in prefs.interests if i in p.tags]
    return 3.0 * min(len(matched), 2) if matched else 0.2


def _score(p: Place, prefs: Preferences, level: str) -> tuple[float, list[str], list[str]]:
    score = interest_points(p, prefs)
    reasons: list[str] = []
    flags: list[str] = []

    matched = [i for i in prefs.interests if i in p.tags]
    if matched:
        reasons.append("matches your interest in " + " and ".join(matched[:2]))

    eg = energy_gap(prefs.energy, p.energy)
    if eg:
        score -= 2.5 * eg
        reasons.append("needs more energy than you have today" if eg > 1 else "a bit more effort than ideal")
    elif p.energy == "low" and prefs.energy == "low":
        score += 0.8
        reasons.append("easy-going pace")

    ng = noise_gap(prefs.social, p.noise)
    if ng:
        score -= 1.2 * ng
    elif p.noise == "quiet" and prefs.social == "quiet":
        score += 0.8
        reasons.append("quiet setting")

    if prefs.avoid_crowds:
        score += {"low": 1.5, "medium": -0.6, "high": -5.0}[level]
        if level == "low":
            reasons.append("low crowds")
    else:
        score += {"low": 0.3, "medium": 0.0, "high": -0.3}[level]

    if p.cost == 0:
        score += 0.4
        reasons.append("free")
    elif prefs.budget_inr:
        score -= min(1.0, p.cost / max(prefs.budget_inr, 1)) * 1.5

    if p.indoor and prefs.indoor_only:
        reasons.append("indoors")

    return score, reasons, flags


def _rank(prefs: Preferences, city: CityData, kind: str) -> OptionsResult:
    have_window = prefs.start_min is not None and prefs.end_min is not None
    start = prefs.start_min if have_window else 10 * 60
    end = prefs.end_min if have_window else 20 * 60
    out: list[Candidate] = []
    dropped: dict[str, int] = {}
    dropped_names: list[tuple[float, str, str]] = []
    pool = [p for p in city.places if p.kind == kind]

    def drop(p: Place, reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1
        hit = sum(1 for i in prefs.interests if i in p.tags)
        dropped_names.append((hit + (0.5 if p.cost == 0 else 0), p.name, reason))

    for p in pool:
        if prefs.avoid_alcohol and p.alcohol_centric:
            drop(p, "built around alcohol")
            continue
        if prefs.indoor_only and kind == "activity" and not p.indoor:
            drop(p, "outdoors")
            continue
        ok, flag = diet_check(p, prefs.diet)
        if not ok:
            drop(p, f"doesn't suit a {prefs.diet} diet")
            continue
        if prefs.budget_inr is not None and p.cost > prefs.budget_inr:
            drop(p, "over your budget")
            continue
        if prefs.minutes is not None and p.min_dur > prefs.minutes:
            drop(p, "too long for your time")
            continue
        if have_window and _open_overlap(p, start, end) < min(p.min_dur, 30):
            drop(p, "no live music during your window" if p.live_only else "closed during your window")
            continue
        level = best_crowd(p, start, end) if have_window else p.crowd
        if prefs.avoid_crowds and level == "high":
            drop(p, "crowded on Saturdays")
            continue
        score, reasons, flags = _score(p, prefs, level)
        if flag:
            flags.append(flag)
        out.append(Candidate(place=p, score=round(score, 2), reasons=reasons, flags=flags))

    out.sort(key=lambda c: (-c.score, c.place.cost, c.place.id))
    dropped_names.sort(key=lambda t: -t[0])
    notable = [{"name": n, "reason": r} for hit, n, r in dropped_names if hit > 0][:4]
    return OptionsResult(candidates=out, dropped=dropped, skipped_notable=notable, pool_size=len(pool))


def get_activity_options(prefs: Preferences, city: CityData) -> OptionsResult:
    return _rank(prefs, city, "activity")


def get_food_options(prefs: Preferences, city: CityData) -> OptionsResult:
    return _rank(prefs, city, "food")


def summarize(res: OptionsResult) -> str:
    n = len(res.candidates)
    if not res.dropped:
        return f"{n} of {res.pool_size} venues kept"
    why = ", ".join(f"{v}x {k}" for k, v in sorted(res.dropped.items(), key=lambda kv: -kv[1])[:3])
    return f"{n} of {res.pool_size} venues kept (removed: {why})"
