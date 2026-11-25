"""Properties every PlanResult must satisfy, whatever went in. Returns violations instead of raising,
so the same checks serve pytest, the nasty-input script and the property sweep."""
from __future__ import annotations

import json
import re

from app.cities import get_city
from app.rules import diet_check
from app.schemas import PlanResult, UserInput
from app.util import parse_windows

STATES = {"running", "ok", "retry", "fallback", "error", "info"}
# words that mean an internal value leaked into text a person reads
LEAKS = [
    (re.compile(r"\bNone\b(?! of)"), "None"), (re.compile(r"\bnan\b"), "nan"), (re.compile(r"undefined|NoneType|Traceback"), "internal text"),
    (re.compile(r"\(s\)"), "(s)"), (re.compile(r"[{}]"), "braces"), (re.compile(r"₹\s?-"), "negative rupees"), (re.compile(r"(?<![\w:])-\d+ min\b"), "negative minutes"),
    (re.compile(r"  +"), "double space"), (re.compile(r"\bTODO\b"), "TODO"),
]


def _texts(res: PlanResult) -> list[tuple[str, str]]:
    out = [("headline", res.headline), ("summary", res.summary), ("message", res.message), ("backup", res.backup or "")]
    out += [("trade_off", t) for t in res.trade_offs] + [("caution", t) for t in res.cautions]
    out += [("assumption", t) for t in res.assumptions] + [("relaxed", t) for t in res.relaxed] + [("question", t) for t in res.questions]
    for s in res.stops:
        out += [(f"{s.id}.why", s.why), (f"{s.id}.note", s.note), (f"{s.id}.verify", s.verify), (f"{s.id}.name", s.name)]
        out += [(f"{s.id}.warning", w) for w in s.warnings] + [(f"{s.id}.flag", f) for f in s.flags]
    return out


def check_result(res: PlanResult, inp: UserInput | None = None) -> list[str]:
    bad: list[str] = []

    def need(cond: bool, msg: str) -> None:
        if not cond:
            bad.append(msg)

    need(res.status in {"ok", "needs_info", "fallback"}, f"unknown status {res.status}")
    need(res.mode in {"ai", "basic"}, f"unknown mode {res.mode}")
    try:
        json.dumps(res.model_dump(), ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        bad.append(f"result is not JSON-serialisable: {e}")

    for where, text in _texts(res):
        for rx, label in LEAKS:
            if text and rx.search(text):
                bad.append(f"{label} in {where}: {text[:90]!r}")

    if res.status == "needs_info":
        need(bool(res.questions) and not res.stops, "needs_info must ask something and plan nothing")
        need(1 <= len(res.questions) <= 2, f"{len(res.questions)} questions")
        return bad

    need(bool(res.stops) and res.totals is not None, "a plan needs stops and totals")
    if not res.stops or res.totals is None:
        return bad
    t = res.totals
    prev_end: int | None = None
    for s in res.stops:
        need(s.end > s.start >= 0 and s.end - s.start == s.duration, f"{s.id}: bad times {s.start}-{s.end} ({s.duration})")
        need(s.end <= 24 * 60, f"{s.id}: runs past midnight")
        need(prev_end is None or s.start >= prev_end + s.travel.minutes, f"{s.id}: starts before the previous stop plus travel")
        need(s.cost >= 0 and s.travel.cost >= 0 and s.travel.minutes >= 0, f"{s.id}: negative cost or travel")
        need(bool(s.name.strip()), "unnamed stop")
        prev_end = s.end
    need(t.spend == sum(s.cost for s in res.stops), "spend does not match the stops")
    need(t.travel_cost == sum(s.travel.cost for s in res.stops), "travel_cost does not match the stops")
    need(t.total_cost == t.spend + t.travel_cost, "total_cost != spend + travel_cost")
    need(t.travel_minutes == sum(s.travel.minutes for s in res.stops), "travel_minutes does not match the stops")
    need(t.start == res.stops[0].start - res.stops[0].travel.minutes or t.start <= res.stops[0].start, "totals.start after the first stop")
    need(t.end == res.stops[-1].end, "totals.end is not the last stop's end")
    if t.budget is not None:
        need(t.remaining == t.budget - t.total_cost, "remaining is not budget - total")

    if res.status == "fallback":
        need(bool(res.message), "a fallback must say why")
        need(res.data_source == "generic" and all(s.id.startswith("generic-") for s in res.stops), "fallback stops must be marked generic")
        need(all(s.verify for s in res.stops), "generic stops must carry a caution")
        return bad

    # ---- a full plan: re-check every hard constraint against the dataset, not against what the agent says
    p = res.preferences
    need(p is not None and bool(p.city), "ok plan without preferences")
    if p is None or not p.city or res.data_source != "curated":
        return bad
    city = get_city(p.city)
    if city is None:
        bad.append(f"no dataset for {p.city}")
        return bad
    by_id = city.by_id()
    ids = [s.id for s in res.stops]
    need(len(set(ids)) == len(ids), "a venue appears twice")
    relaxed_crowds = any("crowded places" in m for m in res.relaxed)
    for s in res.stops:
        venue = by_id.get(s.id)
        if venue is None:
            bad.append(f"{s.id} is not in the {city.key} dataset")
            continue
        need(not s.closed, f"{s.id} is marked closed")
        need(any(a <= s.start and s.end <= b for a, b in parse_windows(venue.hours)), f"{s.id} {s.start}-{s.end} is outside its hours {venue.hours}")
        need(diet_check(venue, p.diet)[0], f"{s.id} breaks the {p.diet} diet")
        need(not (p.avoid_alcohol and venue.alcohol_centric), f"{s.id} is built around alcohol")
        need(not (p.indoor_only and venue.kind == "activity" and not venue.indoor), f"{s.id} is outdoors")
        need(not (p.avoid_crowds and not relaxed_crowds and s.crowd_level == "high"), f"{s.id} is crowded and the user asked to avoid crowds")
        need(s.why.strip() != "", f"{s.id} has no explanation")
        if s.travel.mode == "walk":
            need(s.travel.cost == 0, f"{s.id}: a walk should be free")
    if p.budget_inr is not None:
        need(t.total_cost <= p.budget_inr, f"over budget: {t.total_cost} > {p.budget_inr}")
    if p.end_min is not None:
        need(t.end <= p.end_min, f"over time: ends {t.end}, window ends {p.end_min}")
    need(bool(res.headline.strip()) and bool(res.summary.strip()), "missing headline or summary")
    need(len(res.trade_offs) >= 1, "no trade-off notes")
    return bad


def check_trace(events: list[dict]) -> list[str]:
    bad: list[str] = []
    steps = [e for e in events if e.get("type") == "step"]
    if not steps:
        return ["empty trace"]
    for e in steps:
        if e["state"] not in STATES - {"running"}:
            bad.append(f"step {e['tool']} ended in state {e['state']}")
        if not e["summary"] and e["tool"] not in {"recover"}:
            bad.append(f"step {e['tool']} has no summary")
    ids = [e["id"] for e in steps]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        bad.append("step ids are not unique and increasing")
    return bad
