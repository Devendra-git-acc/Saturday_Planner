"""Tool 4 - estimateCost.

The scheduling simulator: walks through an ordered list of stops and works out when each one
starts and ends, what the travel between them costs in minutes and rupees, whether the venue is
actually open, and how crowded it will be at that moment. The planner and the validator both rely
on this single source of truth, so a plan can never be 'valid' on paper and impossible in practice.
"""
from __future__ import annotations

from .. import geo
from ..rules import crowd_at, diet_check
from ..schemas import CityData, CostReport, DraftStop, Issue, Place, Preferences, ScheduledStop
from ..util import fmt12, parse_windows

MAX_WAIT = 30  # we will wait this long for a venue to open; more than that counts as 'closed'
CLOSED_PREFIX = "Closed: "


def _open_window(p: Place, arrive: int) -> tuple[int, int, int] | None:
    """(start, wait, window_end) for the first window we can still use, or None if all are over."""
    for s, e in parse_windows(p.hours):
        if arrive < e:
            return (arrive, 0, e) if arrive >= s else (s, s - arrive, e)
    return None


def schedule_stop(
    city: CityData,
    prefs: Preferences,
    prev: Place | None,
    t_now: int,
    place: Place,
    want_dur: int | None,
    plan_end: int | None,
) -> ScheduledStop:
    """Place one stop on the clock. Never raises; infeasible stops come back with closed=True."""
    lg = geo.leg(city, prev, place, t_now)
    arrive = t_now + lg.minutes
    warnings: list[str] = []
    closed = False
    wait = 0

    win = _open_window(place, arrive)
    if win is None:
        start, wend, closed = arrive, arrive, True
        warnings.append(f"{CLOSED_PREFIX}already shut by {fmt12(arrive)} (hours {place.hours})")
    else:
        start, wait, wend = win
        if wait > MAX_WAIT:
            closed = True
            warnings.append(f"{CLOSED_PREFIX}does not open until {fmt12(start)} (you would arrive at {fmt12(arrive)})")
            start, wait = arrive, 0

    desired = want_dur or place.duration
    dur = max(place.min_dur, min(place.max_dur, desired))
    if not closed:
        avail = wend - start
        if avail < place.min_dur:
            closed = True
            warnings.append(f"{CLOSED_PREFIX}closes at {fmt12(wend)}, too soon after you would arrive ({fmt12(start)})")
        else:
            dur = min(dur, avail)
            if plan_end is not None and start + dur > plan_end:
                dur = max(place.min_dur, plan_end - start)
            if dur < min(desired, place.max_dur) - 5:
                warnings.append(f"Shortened to {dur} min to fit")

    end = start + dur
    _, flag = diet_check(place, prefs.diet)
    area = city.areas.get(place.area)
    return ScheduledStop(
        id=place.id, name=place.name, kind=place.kind, category=place.category,
        area_name=area.name if area else place.area,
        start=start, end=end, duration=dur, wait=wait, cost=place.cost, travel=lg,
        crowd_level=crowd_at(place, start + dur // 2),  # type: ignore[arg-type]
        closed=closed, energy=place.energy, tags=list(place.tags), note=place.note, verify=place.verify,
        diet=place.diet, flags=[flag] if flag else [], hours=place.hours, source=place.source,
        warnings=warnings,
    )


def estimate_cost(city: CityData, prefs: Preferences, stops: list[DraftStop]) -> CostReport:
    by_id = city.by_id()
    t = prefs.start_min if prefs.start_min is not None else 10 * 60
    start = t
    prev: Place | None = None
    seen: set[str] = set()
    out: list[ScheduledStop] = []
    pre: list[Issue] = []

    for ds in stops:
        p = by_id.get(ds.place_id)
        if p is None:
            pre.append(Issue(code="unknown_stop", severity="error", stop_id=ds.place_id,
                             message=f"'{ds.place_id}' is not a venue in the candidate list - use only candidate ids"))
            continue
        if p.id in seen:
            pre.append(Issue(code="duplicate_stop", severity="error", stop_id=p.id,
                             message=f"{p.name} appears more than once - visit each place once"))
            continue
        seen.add(p.id)
        st = schedule_stop(city, prefs, prev, t, p, ds.duration_min, prefs.end_min)
        out.append(st)
        t, prev = st.end, p

    spend = sum(s.cost for s in out)
    travel_cost = sum(s.travel.cost for s in out)
    end = out[-1].end if out else start
    return CostReport(
        stops=out, spend=spend, travel_cost=travel_cost, total_cost=spend + travel_cost,
        start=start, end=end, total_minutes=end - start,
        travel_minutes=sum(s.travel.minutes for s in out), pre_issues=pre,
    )
