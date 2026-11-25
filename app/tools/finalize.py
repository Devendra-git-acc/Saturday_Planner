"""Tool 7 - generateFinalPlan.

Facts first, words second. Code computes every number and trade-off from the validated schedule;
the model (when available) only phrases them. Its output is then checked: any rupee amount or
clock time that is not in the facts sends that sentence back to the template instead.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..llm import LLM, LLMBadOutput, LLMUnavailable, call_validated
from ..prompts import final_messages, person_view
from ..rules import meals_at
from ..schemas import Candidate, CityData, CostReport, LLMFinal, Preferences
from ..trace import ToolInfo
from ..util import fmt12, fmt_inr, fmt_minutes, plural, short

_REASON_CLAUSES = {
    "crowded on Saturdays": "is usually crowded on Saturdays",
    "closed during your window": "is closed during your time window",
    "no live music during your window": "has no live music during your time window",
    "over your budget": "is over your budget",
    "built around alcohol": "is built around alcohol",
    "outdoors": "is outdoors and you wanted indoor places",
    "too long for your time": "takes longer than your time window",
}


def _clause(reason: str) -> str:
    return _REASON_CLAUSES.get(reason) or (reason if reason.startswith("doesn't") else f"was ruled out ({reason})")


@dataclass
class FinalContent:
    headline: str = ""
    summary: str = ""
    why: dict[str, str] = field(default_factory=dict)
    trade_offs: list[str] = field(default_factory=list)
    backup: str | None = None
    used_llm: bool = False


# ----------------------------------------------------------------- facts


def trade_off_facts(
    prefs: Preferences,
    report: CostReport,
    skipped: list[dict],
    relaxed: list[str],
    missing_interests: list[str],
) -> list[str]:
    facts: list[str] = []
    remaining = (prefs.budget_inr or 0) - report.total_cost
    if report.total_cost == 0:
        facts.append("Everything in this plan is free.")
    else:
        travel_part = f", of which about {fmt_inr(report.travel_cost)} is autos/cabs" if report.travel_cost else ""
        facts.append(
            f"Total about {fmt_inr(report.total_cost)} of your {fmt_inr(prefs.budget_inr or 0)} budget{travel_part}; "
            f"{fmt_inr(max(0, remaining))} left over."
        )
    window = f"The plan runs {fmt12(report.start)} to {fmt12(report.end)}"
    tail = [f"{fmt_minutes(report.travel_minutes)} of it is travel"] if report.travel_minutes else []
    spare = (prefs.end_min or 0) - report.end
    if prefs.end_min is not None and spare >= 60:
        tail.append(f"about {fmt_minutes(spare)} of your window is left open")
    facts.append(window + ("; " + "; ".join(tail) if tail else "") + ".")
    if report.stops:
        longest = max(report.stops, key=lambda s: s.travel.minutes)
        if longest.travel.minutes >= 25:
            facts.append(f"The longest hop is about {longest.travel.minutes} minutes, to {longest.name}.")
    for item in skipped[:2]:
        extra = " and you asked to avoid crowds" if item["reason"] == "crowded on Saturdays" and prefs.avoid_crowds else ""
        facts.append(f"I left out {item['name']} because it {_clause(item['reason'])}{extra}.")
    if missing_interests:
        facts.append("I could not fit " + ", ".join(missing_interests) + " this time without breaking another rule.")
    for r in relaxed:
        facts.append(r)
    return facts


def pick_backups(chosen_ids: set[str], chosen_areas: set[str], acts: list[Candidate], foods: list[Candidate]) -> list[dict]:
    pool = [c for c in [*acts, *foods] if c.place.id not in chosen_ids]
    pool.sort(key=lambda c: (c.place.area not in chosen_areas, -c.score))
    out = []
    for kind in ("activity", "food"):
        c = next((c for c in pool if c.place.kind == kind), None)
        if c:
            out.append({
            "id": c.place.id, "name": c.place.name, "type": kind, "area": c.place.area,
            "near_plan": c.place.area in chosen_areas, "note": c.place.note,
        })
    return out


# -------------------------------------------------------------- template


def _template_why(stop, reasons: list[str], prefs: Preferences) -> str:
    bits: list[str] = []
    first = next((r for r in reasons if r.startswith("matches")), None)
    if first:
        bits.append(first[0].upper() + first[1:])
    if stop.kind == "food":
        meal = sorted(meals_at((stop.start + stop.end) // 2))
        if meal:
            bits.append(f"A sensible {meal[0] if len(meal) == 1 else 'meal'} stop at {fmt12(stop.start)}")
    if prefs.avoid_crowds:
        bits.append(
            "Calm around that time" if stop.crowd_level == "low"
            else "Can get moderately busy, so it is better early in your slot"
        )
    elif prefs.energy == "low" and stop.energy == "low":
        bits.append("An easy pace for a tired day")
    bits.append("Free" if stop.cost == 0 else f"About {fmt_inr(stop.cost)}")
    return ". ".join(bits) + "."


def template_content(
    prefs: Preferences, city_display: str, report: CostReport, reasons: dict[str, list[str]], facts: list[str], backups: list[dict]
) -> FinalContent:
    n = len(report.stops)
    tone = "easy" if prefs.energy == "low" else "full" if prefs.energy == "high" else "relaxed"
    article = "An" if tone[0] in "aeiou" else "A"
    cats: list[str] = []
    for s in report.stops:
        c = (s.category or s.kind).strip()
        if c not in cats:
            cats.append(c)
    cats_txt = ", ".join(cats[:3]) if len(cats) <= 3 else f"{', '.join(cats[:2])} and more"
    names = [s.name for s in report.stops]
    content = FinalContent(
        headline=f"{article} {tone} Saturday in {city_display}: {cats_txt}",
        summary=(
            f"{' → '.join(names)}. About {fmt_inr(report.total_cost)} in all, {fmt12(report.start)} to {fmt12(report.end)}."
        ),
        why={s.id: _template_why(s, reasons.get(s.id, []), prefs) for s in report.stops},
        trade_offs=facts[:4],
    )
    if backups and report.stops:
        alt = backups[0]
        where = "a nearby alternative" if alt.get("near_plan") else "another low-key option"
        content.backup = f"If a stop does not work out, {alt['name']} is {where}."
    return content


# ---------------------------------------------------------------- guard


_MONEY = re.compile(r"(?:₹|rs\.?|inr)\s?([\d,]+)", re.I)
_CLOCK = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s?(am|pm)\b", re.I)


def _numbers(blob: str) -> set[int]:
    return {int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", blob) if x.replace(",", "").isdigit()}


def grounded(text: str, allowed_numbers: set[int], allowed_minutes: set[int]) -> bool:
    for m in _MONEY.findall(text):
        digits = m.replace(",", "")
        if digits.isdigit() and int(digits) not in allowed_numbers:
            return False
    for h, mm, ap in _CLOCK.findall(text):
        hour = int(h) % 12 + (12 if ap.lower() == "pm" else 0)
        t = hour * 60 + int(mm or 0)
        if not any(abs(t - a) <= 1 for a in allowed_minutes):
            return False
    return True


# ---------------------------------------------------------------- entry


def generate_final_plan(
    llm: LLM | None,
    prefs: Preferences,
    city: CityData,
    report: CostReport,
    reasons: dict[str, list[str]],
    facts: list[str],
    backups: list[dict],
    info: ToolInfo,
) -> FinalContent:
    base = template_content(prefs, city.display, report, reasons, facts, backups)
    if llm is None or not report.stops:
        return base

    payload = {
        "city": city.display,
        "person": person_view(prefs),
        "plan": {
            "start": fmt12(report.start), "end": fmt12(report.end),
            "spend_inr": report.spend, "travel_inr": report.travel_cost, "total_inr": report.total_cost,
            "budget_inr": prefs.budget_inr,
        },
        "stops": [
            {
                "id": s.id, "name": s.name, "type": s.kind, "category": s.category, "area": s.area_name,
                "arrive": fmt12(s.start), "leave": fmt12(s.end), "cost_inr": s.cost,
                "travel_before": {"mode": s.travel.mode, "minutes": s.travel.minutes, "cost_inr": s.travel.cost},
                "crowd_then": s.crowd_level, "fit": reasons.get(s.id, [])[:3], "cautions": s.flags,
            }
            for s in report.stops
        ],
        "trade_off_facts": facts,
        "backup_options": backups,
    }
    allowed_n = _numbers(json.dumps(payload, ensure_ascii=False))
    allowed_t = {s.start for s in report.stops} | {s.end for s in report.stops} | {report.start, report.end}

    try:
        out = call_validated(llm, "final", final_messages(payload), LLMFinal, effort="low", info=info)
    except (LLMUnavailable, LLMBadOutput) as e:
        info.fallback = f"template text ({short(e, 90)})"
        info.unavailable = isinstance(e, LLMUnavailable)
        return base

    def ok(text: str, limit: int) -> bool:
        return bool(text and text.strip()) and len(text) <= limit and grounded(text, allowed_n, allowed_t)

    content = FinalContent(used_llm=True)
    content.headline = out.headline.strip() if ok(out.headline, 100) else base.headline
    content.summary = out.summary.strip() if ok(out.summary, 420) else base.summary
    llm_why = {w.id: w.why.strip() for w in out.stops}
    rejected = 0
    for s in report.stops:
        text = llm_why.get(s.id, "")
        if ok(text, 420):
            content.why[s.id] = text
        else:
            content.why[s.id] = base.why[s.id]
            rejected += 1
    tos = [t.strip() for t in out.trade_offs if ok(t, 300)][:4]
    content.trade_offs = tos or base.trade_offs
    names = {b["name"] for b in backups}
    content.backup = out.backup.strip() if out.backup and ok(out.backup, 300) and any(n in out.backup for n in names) else base.backup
    if rejected:
        info.notes.append(f"{plural(rejected, 'sentence')} failed the fact check and used template text")
    return content
