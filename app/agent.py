"""The orchestrator.

The control flow is plain, deterministic Python; the LLM is used at three narrow points (reading
messy preferences, drafting the itinerary, phrasing the explanation) and never gets the last word:
everything it proposes is re-scheduled and re-validated by code.

    parseUserPreferences -> [chooseStartTime] -> getActivityOptions + getFoodOptions
      -> draftPlan (AI) -> estimateCost -> validatePlan  --(problems? repair, max N times)-->
      -> rule-based draft if the AI path never produced a clean plan
      -> relax soft constraints one at a time if nothing valid exists
      -> generic outline as the last resort -> generateFinalPlan
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

from .cities import get_city
from .config import Settings
from .llm import LLM, LLMBadOutput, LLMUnavailable
from .schemas import (
    CityData, CostReport, DraftPlan, Leg, LLMStatus, OptionsResult, PlanResult, Preferences, ScheduledStop,
    Totals, UserInput, ValidationReport,
)
from .tools import osm
from .tools.costing import estimate_cost
from .tools.finalize import generate_final_plan, pick_backups, trade_off_facts
from .tools.options import get_activity_options, get_food_options, summarize
from .tools.planner import choose_start, llm_draft, rule_plan
from .tools.preferences import parse_user_preferences, set_window
from .tools.validate import user_cautions, validate_plan
from .trace import StepHandle, ToolInfo, Tracer
from .util import fmt12, fmt_inr, fmt_minutes, plural, short

log = logging.getLogger("saturday.agent")

DEFAULT_START = 10 * 60


# ------------------------------------------------------------------ helpers


class _Gate:
    """Per-request view of the model: tallies usage and switches off after a hard failure."""

    def __init__(self, llm: LLM | None, deadline: float, reserve: float = 10.0) -> None:
        self.llm, self.deadline, self.reserve = llm, deadline, reserve
        self.down = False
        self.used = False
        self.calls = 0
        self.tokens = 0
        self.notes: list[str] = []

    def usable(self) -> bool:
        return self.llm is not None and not self.down and time.monotonic() < self.deadline - self.reserve

    def get(self) -> LLM | None:
        return self.llm if self.usable() else None

    def absorb(self, info: ToolInfo) -> None:
        self.calls += info.llm_calls
        self.tokens += info.tokens
        if info.llm_calls and not info.fallback:
            self.used = True
        self.notes.extend(info.notes)
        if info.fallback:
            self.notes.append(info.fallback)
        if info.unavailable:
            self.down = True

    def fail(self, e: Exception) -> None:
        if isinstance(e, LLMUnavailable):
            self.down = True
        self.notes.append(short(e, 120))


def _status(info: ToolInfo) -> str:
    return "fallback" if info.fallback else "retry" if info.retries else "ok"


def _describe_input(inp: UserInput) -> str:
    bits = []
    if inp.city:
        bits.append(inp.city)
    if inp.budget is not None:
        bits.append(fmt_inr(inp.budget))
    if inp.available_time:
        bits.append(inp.available_time)
    if inp.interests:
        bits.append(f"{len(inp.interests)} interests")
    if inp.constraints:
        bits.append(f"{len(inp.constraints)} constraints")
    if inp.mood:
        bits.append("mood")
    if inp.free_text:
        bits.append(f"free text ({len(inp.free_text)} chars)")
    return ", ".join(bits) or "(empty)"


def _prefs_summary(p: Preferences) -> str:
    bits = [p.city_display or "city?"]
    bits.append(fmt_inr(p.budget_inr or 0))
    bits.append(fmt_minutes(p.minutes or 0))
    bits.append(f"energy {p.energy}")
    if p.interests:
        bits.append("likes " + ", ".join(p.interests))
    flags = [x for x, on in (("avoid crowds", p.avoid_crowds), ("no alcohol", p.avoid_alcohol), ("indoor only", p.indoor_only)) if on]
    if p.diet != "none":
        flags.insert(0, p.diet)
    if flags:
        bits.append(", ".join(flags))
    return "; ".join(bits)


@dataclass
class _Eval:
    report: CostReport
    validation: ValidationReport
    prefs: Preferences  # the (possibly relaxed) preferences this was judged against
    source: str  # ai | rules

    @property
    def key(self) -> tuple:
        return (self.validation.ok, -self.validation.penalty)


# ------------------------------------------------------------------ agent


class Agent:
    def __init__(self, settings: Settings, llm: LLM | None) -> None:
        self.s = settings
        self.llm = llm

    # public entry point ------------------------------------------------
    def run(self, inp: UserInput, tracer: Tracer, *, allow_llm: bool = True) -> PlanResult:
        t0 = time.monotonic()
        gate = _Gate(self.llm if allow_llm else None, t0 + self.s.request_deadline_s)
        try:
            res = self._run(inp, tracer, gate)
        except Exception:  # noqa: BLE001 - the user must always get a useful answer
            log.exception("agent crashed; returning generic outline")
            tracer.note("recover", "Unexpected internal error; switched to a generic outline", status="fallback")
            prefs = Preferences(city_display=inp.city, budget_inr=int(inp.budget) if inp.budget else 1500, minutes=240)
            res = self._generic(prefs, inp.city or "your city", "Something unexpected went wrong on my side.", tracer)
        res.llm = LLMStatus(
            enabled=self.llm is not None, used=gate.used, model=getattr(self.llm, "model", None),
            calls=gate.calls, tokens=gate.tokens, notes=list(dict.fromkeys(gate.notes))[:6],
        )
        res.mode = "ai" if gate.used else "basic"
        res.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return res

    # main flow ---------------------------------------------------------
    def _run(self, inp: UserInput, tracer: Tracer, gate: _Gate) -> PlanResult:
        # 1 ─ understand the request
        with tracer.step("parseUserPreferences", inp=_describe_input(inp)) as st:
            info = ToolInfo()
            prefs = parse_user_preferences(inp, gate.get(), info)
            gate.absorb(info)
            st.status = _status(info)
            how = "model + rules" if prefs.parsed_by == "llm" else "rules"
            st.summary = f"{_prefs_summary(prefs)}  [{how}]"
            st.detail = {"assumptions": prefs.assumptions, "notes": prefs.notes, "other_constraints": prefs.other_constraints}

        if not prefs.city:
            return self._needs_info(prefs, tracer)

        # 2 ─ which data do we have for this city?
        city: CityData | None = get_city(prefs.city) if prefs.city_known else None
        source = "curated"
        tried_live = city is None and self.s.enable_live_data
        if tried_live:
            with tracer.step("lookupLiveVenues", inp=prefs.city_display or prefs.city) as st:
                city = osm.fetch_city(prefs.city_display or prefs.city, contact=self.s.contact_email)
                if city:
                    source = "openstreetmap"
                    st.summary = f"{len(city.places)} venues near the centre of {city.display} (OpenStreetMap)"
                else:
                    st.status = "fallback"
                    st.summary = f"no usable live data for '{prefs.city_display}'"
        if city is None:
            name = prefs.city_display or prefs.city
            why = f"I don't have venue data for {name} yet (curated cities: Bangalore, Pune, Gurugram)."
            if tried_live:
                why = (
                    f"I don't have venue data for {name} yet. The live OpenStreetMap lookup returned nothing usable "
                    "(the place may not be on the map, or its free public servers are busy), so check the spelling, "
                    "try again in a minute, or pick Bangalore, Pune or Gurugram."
                )
            return self._generic(prefs, name, why, tracer)

        # 3 ─ pick a start time if none was given
        if prefs.start_min is None:
            with tracer.step("chooseStartTime", inp="no start time given") as st:
                a0, f0 = get_activity_options(prefs, city), get_food_options(prefs, city)
                start, scores = choose_start(prefs, city, a0.candidates, f0.candidates)
                set_window(prefs, start)
                if scores:
                    cmp_txt = ", ".join(f"{fmt12(k)}={v}" for k, v in scores.items())
                    st.summary = f"compared {cmp_txt} → {fmt12(start)}"
                    prefs.assumptions.append(f"No start time given, so I compared a few and chose {fmt12(start)} as the best fit.")
                else:
                    st.status = "fallback"
                    st.summary = f"nothing fits cleanly; defaulting to {fmt12(start)}"
                    prefs.assumptions.append(f"No start time given; I assumed {fmt12(start)}.")

        # 4 ─ candidate venues under the hard constraints
        with tracer.step("getActivityOptions") as st:
            acts = get_activity_options(prefs, city)
            st.summary = summarize(acts)
            st.detail = {"top": [c.place.name for c in acts.candidates[:5]], "dropped": acts.dropped}
        with tracer.step("getFoodOptions") as st:
            foods = get_food_options(prefs, city)
            st.summary = summarize(foods)
            st.detail = {"top": [c.place.name for c in foods.candidates[:5]], "dropped": foods.dropped}

        # 5 ─ draft → cost → validate (→ repair)
        chosen = self._plan(prefs, city, acts, foods, tracer, gate)
        relaxed_msgs: list[str] = []
        work = prefs
        if chosen is None or not chosen.validation.ok:
            chosen, work, acts, foods, relaxed_msgs = self._relax(prefs, city, tracer, chosen)
        if chosen is None or not chosen.validation.ok or not chosen.report.stops:
            why = self._why_nothing(prefs, acts, foods)
            return self._generic(prefs, city.display, why, tracer, source=source)

        return self._finish(inp, prefs, work, city, source, chosen, acts, foods, relaxed_msgs, tracer, gate)

    # planning loop -----------------------------------------------------
    def _evaluate(self, tracer: Tracer, city: CityData, prefs: Preferences, draft: DraftPlan, foods: OptionsResult, source: str, *, final: bool) -> _Eval:
        with tracer.step("estimateCost", inp=plural(len(draft.stops), "stop")) as st:
            report = estimate_cost(city, prefs, draft.stops)
            st.summary = (
                f"{fmt12(report.start)}–{fmt12(report.end)}, {fmt_inr(report.total_cost)} "
                f"(venues {fmt_inr(report.spend)} + travel {fmt_inr(report.travel_cost)}), {report.travel_minutes} min travelling"
            )
        with tracer.step("validatePlan") as st:
            v = validate_plan(report, prefs, city, food_available=bool(foods.candidates))
            errs, majors = v.errors, [i for i in v.warnings if i.major]
            shown = "; ".join(i.message for i in v.repair_issues[:3])
            if v.clean:
                st.summary = "all hard constraints pass" + (f"; {plural(len(v.warnings), 'minor note')}" if v.warnings else "")
            elif errs:
                st.summary = f"{plural(len(errs), 'hard-constraint error')}: {short(shown, 220)}"
                st.status = "error" if final else "retry"
            else:
                st.summary = f"valid, but {plural(len(majors), 'thing')} could be better: {short(shown, 200)}"
                st.status = "ok" if final else "retry"
            st.detail = [{"severity": i.severity, "code": i.code, "message": i.message} for i in v.issues][:12]
        return _Eval(report, v, prefs, source)

    def _plan(self, prefs: Preferences, city: CityData, acts: OptionsResult, foods: OptionsResult, tracer: Tracer, gate: _Gate) -> _Eval | None:
        if not acts.candidates and not foods.candidates:
            tracer.note("draftPlan", "no candidate venues survived the hard constraints", status="error")
            return None

        best: _Eval | None = None
        configured = gate.llm is not None
        if gate.usable():
            problems: list[str] | None = None
            prev: CostReport | None = None
            for k in range(self.s.max_repairs + 1):
                if not gate.usable():
                    break
                label = "draftPlan (AI)" if k == 0 else f"repairPlan #{k} (AI)"
                draft: DraftPlan | None = None
                with tracer.step("draftPlan", label=label, inp=f"{plural(len(acts.candidates), 'activity', 'activities')} and {plural(len(foods.candidates), 'food option')}") as st:
                    info = ToolInfo()
                    try:
                        draft = llm_draft(gate.llm, prefs, city, acts.candidates, foods.candidates, info, previous=prev, problems=problems)  # type: ignore[arg-type]
                        st.summary = " → ".join(self._name(city, s.place_id) for s in draft.stops)
                        st.status = "retry" if info.retries else "ok"
                    except (LLMUnavailable, LLMBadOutput) as e:
                        gate.fail(e)
                        st.status = "fallback"
                        st.summary = f"model unavailable or unusable: {short(e, 140)}"
                    gate.absorb(info)
                if draft is None:
                    break
                ev = self._evaluate(tracer, city, prefs, draft, foods, "ai", final=k == self.s.max_repairs)
                if best is None or ev.key > best.key:
                    best = ev
                if ev.validation.clean:
                    break
                problems = [i.message for i in ev.validation.repair_issues][:8]
                prev = ev.report

        if best is not None and best.validation.clean:
            return best

        # rule-based draft: the safety net, and a yardstick for the AI plan
        label = "draftPlan (rule-based)"
        with tracer.step("draftPlan", label=label, inp="deterministic beam search") as st:
            rp = rule_plan(prefs, city, acts.candidates, foods.candidates)
            if rp is None:
                st.status = "error"
                st.summary = "no schedule satisfies every hard constraint"
            else:
                st.summary = " → ".join(self._name(city, s.place_id) for s in rp.stops)
                if configured and best is None:
                    st.status = "fallback"
                    st.summary += "  (the AI step was unavailable, so this is basic mode)"
        if rp is not None:
            ev = self._evaluate(tracer, city, prefs, rp, foods, "rules", final=True)
            if best is None or ev.key > best.key:
                best = ev
        return best

    def _relax(self, prefs: Preferences, city: CityData, tracer: Tracer, current: _Eval | None):
        """Nothing valid: loosen SOFT preferences one at a time (never diet, budget, hours or time)."""
        work = prefs.model_copy(deep=True)
        msgs: list[str] = []
        steps: list[tuple[str, Callable[[Preferences], None], str]] = []
        if prefs.avoid_crowds:
            steps.append(("avoid crowds", lambda p: setattr(p, "avoid_crowds", False),
                          "I relaxed 'avoid crowded places' because nothing calm fit your other limits, so check the busy-time notes."))
        steps.append(("energy and noise limits", lambda p: (setattr(p, "energy", "high"), setattr(p, "social", "lively")),
                      "I eased the energy and noise preferences to find anything that fit."))
        if prefs.interests:
            steps.append(("interest matching", lambda p: setattr(p, "interests", []),
                          "I stopped insisting on your listed interests to find anything that fit."))
        acts, foods = get_activity_options(work, city), get_food_options(work, city)
        best = current
        for name, apply, msg in steps:
            apply(work)
            acts, foods = get_activity_options(work, city), get_food_options(work, city)
            with tracer.step("relaxConstraint", inp=name) as st:
                st.status = "fallback"
                st.summary = f"relaxed {name}; {plural(len(acts.candidates), 'activity', 'activities')} and {plural(len(foods.candidates), 'food option')} now"
            msgs.append(msg)
            if not acts.candidates and not foods.candidates:
                continue
            with tracer.step("draftPlan", label="draftPlan (rule-based, relaxed)", inp=name) as st:
                rp = rule_plan(work, city, acts.candidates, foods.candidates)
                st.summary = " → ".join(self._name(city, s.place_id) for s in rp.stops) if rp else "still no valid schedule"
                if rp is None:
                    st.status = "error"
            if rp is None:
                continue
            ev = self._evaluate(tracer, city, work.model_copy(deep=True), rp, foods, "rules", final=True)
            if ev.validation.ok and ev.report.stops:
                return ev, work, acts, foods, list(msgs)
            best = ev if (best is None or ev.key > best.key) else best
        return best, work, acts, foods, list(msgs)

    @staticmethod
    def _name(city: CityData, pid: str) -> str:
        p = city.by_id().get(pid)
        return p.name if p else pid

    @staticmethod
    def _why_nothing(prefs: Preferences, acts: OptionsResult, foods: OptionsResult) -> str:
        drops: dict[str, int] = {}
        for d in (acts.dropped, foods.dropped):
            for k, v in d.items():
                drops[k] = drops.get(k, 0) + v
        if drops:
            top = ", ".join(f"{v} {k}" for k, v in sorted(drops.items(), key=lambda kv: -kv[1])[:3])
            return (
                f"Nothing in my data fits all of your limits ({fmt_inr(prefs.budget_inr or 0)}, {fmt_minutes(prefs.minutes or 0)}"
                f"{', ' + prefs.diet if prefs.diet != 'none' else ''}). Venues removed: {top}."
            )
        return "I could not build a schedule that satisfies all of your limits."

    # finishing ---------------------------------------------------------
    def _finish(self, inp, prefs, work, city, source, ev: _Eval, acts, foods, relaxed_msgs, tracer, gate) -> PlanResult:
        report, v = ev.report, ev.validation
        reasons = {c.place.id: c.reasons for c in [*acts.candidates, *foods.candidates]}
        covered = {t for s in report.stops for t in s.tags}
        missing = [i for i in prefs.interests if i not in covered]
        skipped = [*acts.skipped_notable, *foods.skipped_notable]
        skipped.sort(key=lambda d: 0 if d["reason"] == "crowded on Saturdays" else 1)
        in_data = {t for p in city.places for t in p.tags}
        facts = trade_off_facts(prefs, report, skipped, relaxed_msgs, missing,
                                [i for i in missing if i not in in_data], city.display)
        backups = pick_backups({s.id for s in report.stops}, {city.by_id()[s.id].area for s in report.stops}, acts.candidates, foods.candidates)

        with tracer.step("generateFinalPlan", inp=f"{plural(len(report.stops), 'validated stop')}") as st:
            info = ToolInfo()
            content = generate_final_plan(gate.get(), prefs, city, report, reasons, facts, backups, info)
            gate.absorb(info)
            st.status = _status(info)
            st.summary = ("explanations written by the model and fact-checked" if content.used_llm else "explanations from templates") \
                + f"; {plural(len(content.trade_offs), 'trade-off note')}"

        stops: list[ScheduledStop] = []
        for s in report.stops:
            s = s.model_copy(deep=True)
            s.why = content.why.get(s.id, "")
            if prefs.avoid_crowds and s.crowd_level != "low":
                s.warnings.append(
                    f"Can be {'crowded' if s.crowd_level == 'high' else 'moderately busy'} around {fmt12((s.start + s.end) // 2)}"
                )
            stops.append(s)

        t = Totals(
            spend=report.spend, travel_cost=report.travel_cost, total_cost=report.total_cost, budget=prefs.budget_inr,
            remaining=(prefs.budget_inr - report.total_cost) if prefs.budget_inr is not None else None,
            start=report.start, end=report.end, minutes_available=prefs.minutes or 0,
            minutes_used=report.total_minutes, travel_minutes=report.travel_minutes,
        )
        cautions = user_cautions(v)
        # needs the data cannot answer (wheelchair access, halal, a toddler...) are never checked, so say so
        cautions += [f"I can't check \"{c}\" against my venue data, so please confirm it with each place."
                     for c in prefs.other_constraints]
        return PlanResult(
            status="ok", headline=content.headline, summary=content.summary, city=city.display, data_source=source,
            stops=stops, totals=t, trade_offs=content.trade_offs, backup=content.backup, cautions=cautions,
            assumptions=[*prefs.assumptions, *prefs.notes], relaxed=relaxed_msgs,
            questions=self._questions(prefs, inp), preferences=prefs,
        )

    # non-plan outcomes -------------------------------------------------
    @staticmethod
    def _questions(prefs: Preferences, inp: UserInput | None = None, *, city_missing: bool = False) -> list[str]:
        qs: list[str] = []
        if city_missing:
            qs.append("Which city should I plan for? I have curated data for Bangalore, Pune and Gurugram, and can try others.")
        if "budget" in prefs.defaulted:
            qs.append(f"Roughly what budget do you have? I assumed {fmt_inr(prefs.budget_inr or 0)} for the whole outing.")
        if "time" in prefs.defaulted:
            qs.append("How much time do you have, and when would you like to start? I assumed 4 hours.")
        if "interests" in prefs.defaulted:
            qs.append("What are you in the mood for: food, music, walks, art, history, books, shopping...?")
        has_rules = prefs.diet != "none" or prefs.avoid_crowds or prefs.avoid_alcohol or prefs.indoor_only or prefs.other_constraints
        if not has_rules and len(qs) < 2:
            qs.append("Any dietary needs (vegetarian, vegan, Jain) or things to avoid, like crowds or alcohol?")
        return qs[:2]

    def _needs_info(self, prefs: Preferences, tracer: Tracer) -> PlanResult:
        with tracer.step("askClarifyingQuestions", inp="no city given") as st:
            qs = self._questions(prefs, city_missing=True)
            st.status = "info"
            st.summary = f"{plural(len(qs), 'question')}: " + " / ".join(qs)
        return PlanResult(
            status="needs_info", mode="basic", message="I need one detail before I can plan well.", questions=qs,
            assumptions=[*prefs.assumptions, *prefs.notes], preferences=prefs,
        )

    def _generic(self, prefs: Preferences, city_display: str, reason: str, tracer: Tracer, source: str = "generic") -> PlanResult:
        with tracer.step("fallbackPlan", inp=reason[:120]) as st:
            st.status = "fallback"
            start = prefs.start_min if prefs.start_min is not None else DEFAULT_START
            minutes = prefs.minutes or 240
            budget = prefs.budget_inr if prefs.budget_inr is not None else 1500
            diet = "" if prefs.diet == "none" else f"{prefs.diet} "
            plan: list[tuple[str, str, str, str, int, str]] = [
                ("activity", "walk", "A park or lakeside walk near you", "Search maps for 'park' or 'lake' close to you; go when it is calm (early or near sunset).", 0, "A free, easy way to start and to decompress."),
                ("food", "meal", f"A well-reviewed {diet}cafe or restaurant near you", "Pick somewhere with good recent reviews that is not packed at that hour.", min(450, int(budget * 0.45) // 50 * 50), "A proper sit-down break in the middle."),
                ("activity", "browse", "A bookshop, gallery or small museum nearby", "Check opening hours first; many close on Mondays.", min(150, max(0, budget // 10 // 50 * 50)), "A quiet, low-effort finish."),
            ]
            if minutes < 90:
                plan = plan[:1]
            elif minutes < 180:
                plan = plan[:2]
            share = [0.35, 0.4, 0.25][: len(plan)]
            share = [x / sum(share) for x in share]
            t = start
            stops: list[ScheduledStop] = []
            for i, ((kind, cat, name, note, cost, why), frac) in enumerate(zip(plan, share)):
                dur = max(25, int(minutes * frac) // 5 * 5 - (10 if i else 0))
                begin = t + (10 if i else 0)
                stops.append(ScheduledStop(
                    id=f"generic-{i + 1}", name=name, kind=kind, category=cat, area_name="Near you", start=begin, end=begin + dur,
                    duration=dur, cost=cost, travel=Leg(minutes=10 if i else 0, cost=0, mode="walk/auto" if i else "start"),
                    crowd_level="medium", energy="low", note=note, source="generic", why=why,
                    verify=f"Generic suggestion - I have no venue data for {city_display}.",
                    tags=["walks", "nature"] if cat == "walk" else ["food"] if kind == "food" else ["books", "art"],
                ))
                t = begin + dur
            total = sum(s.cost for s in stops)
            st.summary = f"{plural(len(stops), 'generic stop')}, about {fmt_inr(total)}"
        return PlanResult(
            status="fallback", headline=f"A flexible Saturday outline for {city_display}",
            summary="I could not build a venue-specific plan, so here is a generic outline you can adapt.",
            city=city_display, data_source="generic", stops=stops,
            totals=Totals(spend=total, total_cost=total, budget=prefs.budget_inr, remaining=(budget - total),
                          start=start, end=stops[-1].end if stops else start, minutes_available=minutes,
                          minutes_used=(stops[-1].end - start) if stops else 0,
                          travel_minutes=sum(s.travel.minutes for s in stops)),
            message=reason, trade_offs=["These are categories, not specific venues, so prices and hours are rough guesses."],
            assumptions=[*prefs.assumptions, *prefs.notes], preferences=prefs,
            questions=["Try Bangalore, Pune or Gurugram for a fully detailed plan, or tell me a nearby area to search."],
        )
