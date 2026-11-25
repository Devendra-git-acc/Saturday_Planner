"""Tool 5 - validatePlan.

Everything the LLM (or any planner) proposes is re-checked here against the data, never against
what the model claims. Errors are hard-constraint violations; 'major' warnings are quality
problems worth one more attempt; the rest are shown to the user as cautions.
"""
from __future__ import annotations

from ..rules import diet_check, energy_gap, meal_ok, noise_gap, window_covers_meal, MEAL_WINDOWS
from ..schemas import CityData, CostReport, Issue, Preferences, ValidationReport
from ..util import fmt12, fmt_inr, overlap_minutes
from .costing import CLOSED_PREFIX

LONG_HOP_WARN = 45
LONG_HOP_ERR = 75
ERROR_W, MAJOR_W, WARN_W = 100.0, 12.0, 3.0


def validate_plan(
    report: CostReport, prefs: Preferences, city: CityData, *, food_available: bool = True
) -> ValidationReport:
    issues: list[Issue] = list(report.pre_issues)
    by_id = city.by_id()
    stops = report.stops

    def add(code: str, sev: str, msg: str, stop: str | None = None, major: bool = False, user: str | None = None) -> None:
        issues.append(Issue(code=code, severity=sev, message=msg, stop_id=stop, major=major, user_message=user))  # type: ignore[arg-type]

    if not stops:
        add("empty", "error", "The plan has no usable stops")

    # ---- per stop: hard constraints
    for s in stops:
        p = by_id[s.id]
        closed_msg = next((w for w in s.warnings if w.startswith(CLOSED_PREFIX)), None)
        if s.closed:
            add("closed", "error", f"{s.name}: {(closed_msg or 'closed').removeprefix(CLOSED_PREFIX)}", s.id)
        if not diet_check(p, prefs.diet)[0]:
            add("diet", "error", f"{s.name} does not suit a {prefs.diet} diet - replace it", s.id)
        if prefs.avoid_alcohol and p.alcohol_centric:
            add("alcohol", "error", f"{s.name} is built around alcohol - replace it", s.id)
        if prefs.indoor_only and p.kind == "activity" and not p.indoor:
            add("indoor", "error", f"{s.name} is outdoors but you want indoor places", s.id)
        if prefs.avoid_crowds and s.crowd_level == "high":
            add("crowded", "error",
                f"{s.name} is usually crowded around {fmt12((s.start + s.end) // 2)} and you asked to avoid crowds "
                "- swap it or visit at a calmer time", s.id)
        if s.travel.minutes > LONG_HOP_ERR:
            add("long_hop", "error", f"Getting to {s.name} takes about {s.travel.minutes} min - too far for this outing", s.id)

    # ---- per stop: quality
    low = prefs.energy == "low"
    heavy = 0
    for i, s in enumerate(stops):
        p = by_id[s.id]
        gap = energy_gap(prefs.energy, p.energy)
        if gap >= 2:
            add("energy", "warn", f"{s.name} is demanding and you are low on energy", s.id, major=True)
        elif gap == 1 and low:
            add("energy", "warn", f"{s.name} takes a little more effort than ideal for a tired day", s.id)
        if p.energy != "low":
            heavy += 1
        if noise_gap(prefs.social, p.noise) >= 2:
            add("noise", "warn", f"{s.name} is loud and you prefer quiet", s.id, major=True)
        if LONG_HOP_WARN < s.travel.minutes <= LONG_HOP_ERR:
            add("long_hop", "warn", f"Getting to {s.name} takes about {s.travel.minutes} min", s.id, major=True)
        if p.kind == "food" and not meal_ok(p, s.start, s.end):
            add("meal_time", "warn", f"{s.name} is better visited at a different time of day than {fmt12(s.start)}", s.id)
        if s.wait >= 15:
            add("wait", "warn", f"You would wait about {s.wait} min for {s.name} to open", s.id)
        if i and stops[i - 1].kind == "food" and s.kind == "food" and min(s.duration, stops[i - 1].duration) >= 30:
            add("back_to_back_food", "warn", f"Two sit-down food stops in a row ({stops[i - 1].name}, {s.name})", s.id)
        if any(w.startswith("Shortened") for w in s.warnings):
            add("shortened", "info", f"{s.name}: " + next(w for w in s.warnings if w.startswith("Shortened")), s.id)
    if low and heavy > 2:
        add("pacing", "warn", "Too many effortful stops for a tired day", None, major=True)

    # ---- whole plan
    if prefs.budget_inr is not None and report.total_cost > prefs.budget_inr:
        over = report.total_cost - prefs.budget_inr
        add("over_budget", "error",
            f"Total {fmt_inr(report.total_cost)} (venues {fmt_inr(report.spend)} + travel {fmt_inr(report.travel_cost)}) "
            f"exceeds the budget of {fmt_inr(prefs.budget_inr)} by {fmt_inr(over)} - swap or drop a paid stop")
    if prefs.end_min is not None and stops and report.end > prefs.end_min:
        add("over_time", "error",
            f"The plan runs until {fmt12(report.end)}, {report.end - prefs.end_min} min past your window "
            f"(which ends {fmt12(prefs.end_min)}) - drop a stop or shorten durations")

    n, total, minutes = len(stops), max(1, report.total_minutes), prefs.minutes or 240
    if stops and report.travel_minutes / total > 0.35:
        add("travel_heavy", "warn",
            f"{report.travel_minutes} of {report.total_minutes} minutes are spent travelling - group stops by area", None, major=True,
            user=f"{report.travel_minutes} of the {report.total_minutes} minutes go on travelling between places.")
    cap = 5 if low else 6
    if n > cap:
        add("too_many", "warn", f"{n} stops feels rushed - keep it to {cap} or fewer", None, major=True,
            user=f"{n} stops may feel rushed.")
    if stops and n < 2 and minutes >= 150:
        add("too_few", "warn", "Only one stop for a long outing - add something", None, major=True,
            user="Only one stop fit within your limits, so much of your time window is free.")
    if stops and minutes >= 120 and report.total_minutes < 0.6 * minutes:
        add("underfilled", "warn",
            f"Only {report.total_minutes} of your {minutes} minutes are used - there is room for another stop", None, major=True,
            user=f"Only {report.total_minutes} of your {minutes} minutes are planned, so you have spare time.")

    if food_available and stops and prefs.start_min is not None and prefs.end_min is not None:
        for meal in ("lunch", "dinner"):
            if meal not in window_covers_meal(prefs.start_min, prefs.end_min):
                continue
            ws, we = MEAL_WINDOWS[meal]
            if overlap_minutes(prefs.start_min, prefs.end_min, ws, we) < 90:
                continue
            if not any(s.kind == "food" and s.start < we and s.end > ws for s in stops):
                add("meal_gap", "warn", f"Your window covers {meal} time but no meal is planned then", None, major=True)

    covered = {t for s in stops for t in s.tags}
    missing = [i for i in prefs.interests if i not in covered]
    if prefs.interests and stops and len(missing) == len(prefs.interests):
        add("interests", "warn", "None of your interests are covered: " + ", ".join(prefs.interests), None, major=True)
    elif missing and stops:
        add("interests", "info", "Not covered this time: " + ", ".join(missing))

    # ---- scoring for 'which attempt is best'
    errors = sum(1 for i in issues if i.severity == "error")
    majors = sum(1 for i in issues if i.severity == "warn" and i.major)
    minors = sum(1 for i in issues if i.severity == "warn" and not i.major)
    fill_gap = max(0.0, 1.0 - report.total_minutes / minutes) if minutes else 0.0
    uncovered = len(missing) if stops else 0
    penalty = (errors * ERROR_W + majors * MAJOR_W + minors * WARN_W + 0.05 * report.travel_minutes
               + 8.0 * fill_gap + 4.0 * uncovered)
    return ValidationReport(ok=errors == 0, clean=errors == 0 and majors == 0, issues=issues, penalty=round(penalty, 2))


def user_cautions(v: ValidationReport, limit: int = 5) -> list[str]:
    """Warnings worded for the person reading the plan (not for the planner), without repeats."""
    warns = [i for i in v.issues if i.severity == "warn"]
    codes = {i.code for i in warns}
    out: list[str] = []
    for i in warns:
        if i.code == "underfilled" and "too_few" in codes:
            continue  # the single-stop note already says it
        out.append(i.user_message or i.message)
    return list(dict.fromkeys(out))[:limit]
