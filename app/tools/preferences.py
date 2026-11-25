"""Tool 1 - parseUserPreferences.

Rules extract what they can deterministically (so the app works with no LLM and so hard constraints
are never lost). The LLM, when available, reads the soft, messy part ("tired but wants to do
something fun") and the merge is conservative: the model may ADD constraints, never remove them.
"""
from __future__ import annotations

import re

from ..cities import match_city
from ..llm import LLM, LLMBadOutput, LLMUnavailable, call_validated
from ..prompts import parse_messages
from ..rules import normalize_interests
from ..schemas import INTEREST_VOCAB, LLMPrefs, Preferences, UserInput
from ..trace import ToolInfo
from ..util import clean_label, fmt12, fmt_inr, parse_clock, parse_clock_range, parse_duration_minutes, parse_money, short

DEFAULT_BUDGET = 1500
DEFAULT_MINUTES = 240
MIN_MINUTES, MAX_MINUTES = 30, 720
LATEST_START, LAST_MINUTE = 22 * 60, 23 * 60 + 59  # plans start by 10 pm and end by 11:59 pm
_DIET_RANK = {"none": 0, "vegetarian": 1, "jain": 2, "vegan": 2}

# ---------------------------------------------------------------- keyword rules

_LOW_ENERGY = re.compile(r"\b(tired|exhausted|drained|sleepy|lazy|burn(?:t|ed)[- ]?out|worn out|low[- ]energy|relax(?:ed|ing)?|chill|slow|easy[- ]?going|rest)\b")
_HIGH_ENERGY = re.compile(r"\b(energetic|adventurous|active|pumped|hyper|high[- ]energy|excited|full of energy|sporty|restless)\b")
_QUIET = re.compile(r"\b(quiet|peaceful|calm|serene|solitude|alone|introvert\w*|low[- ]key|away from (?:the )?(?:noise|crowds?))\b")
_LIVELY = re.compile(r"\b(lively|social|party|buzz\w*|vibrant|energetic crowd|going out|fun crowd)\b")

_NONVEG = re.compile(r"\bnon[- ]?veg(?:etarian)?\b")
_VEGAN = re.compile(r"\bvegan\b")
_JAIN = re.compile(r"\bjain\b")
_EGGETARIAN = re.compile(r"\beggetarian\b")
_VEG = re.compile(r"\b(pure[- ]?veg|vegetarian|veggie|veg(?:\s+only)?|no meat|meat[- ]free|no non[- ]?veg)\b")
_CROWD = re.compile(
    r"(avoid\w*\s+(?:the\s+)?(?:crowd\w*|busy|rush\w*|packed)|no\s+crowds?|not\s+(?:too\s+)?crowded|"
    r"crowd[- ]?free|uncrowded|less\s+crowd\w*|away from (?:the )?crowds?|crowded\s+places?\s+(?:are\s+)?(?:a\s+)?no)"
)
_ALCOHOL = re.compile(r"\b(no\s+alcohol|alcohol[- ]?free|non[- ]?alcoholic|teetotal\w*|dry\s+(?:day|outing|place)|no\s+(?:drinking|pubs?|bars?)|don'?t\s+drink|do not drink)\b")
_INDOOR = re.compile(r"\b(indoors?(?:\s+only)?|air[- ]?condition\w*|avoid\s+(?:the\s+)?(?:heat|sun|rain)|rainy|too\s+hot)\b")
_CUE_START = re.compile(r"\b(?:start(?:ing)?|begin(?:ning)?|leav\w+|head(?:ing)? out|go(?:ing)? out|from|after|around|at)\b")


def _has(rx: re.Pattern[str], text: str) -> bool:
    return bool(rx.search(text))


def _diet_from(text: str) -> str:
    t = _NONVEG.sub(" ", text)
    if _VEGAN.search(t):
        return "vegan"
    if _JAIN.search(t):
        return "jain"
    if _VEG.search(t) or _EGGETARIAN.search(t):
        return "vegetarian"
    return "none"


def _interests_from_text(text: str) -> list[str]:
    from ..rules import INTEREST_SYNONYMS

    found: list[str] = []
    lowered = text.lower()
    for word in sorted([*INTEREST_VOCAB, *INTEREST_SYNONYMS], key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            mapped = word if word in INTEREST_VOCAB else INTEREST_SYNONYMS[word]
            if mapped not in found:
                found.append(mapped)
    return found


# ---------------------------------------------------------------- rules parse


def rules_parse(inp: UserInput) -> Preferences:
    p = Preferences(parsed_by="rules")
    free = inp.free_text or ""
    mood = inp.mood or ""
    cons = list(inp.constraints)
    text = " ".join([mood, free, " ".join(cons)]).lower()

    # city
    given = inp.city
    city, how = match_city(given) if given else match_city(free, fuzzy=False)
    if city:
        p.city, p.city_display, p.city_known = city.key, city.display, True
        p.city_input = given or city.display
        if how == "fuzzy":
            p.assumptions.append(f"I read '{short(given or '', 40)}' as {city.display}.")
    elif how == "ambiguous":
        p.city_input = given
        p.city_ambiguous = True
        p.assumptions.append("You named more than one city, so I need you to pick one.")
    elif given and clean_label(given):
        label = clean_label(given)
        p.city_input = label
        p.city, p.city_display, p.city_known = label.lower(), label.title(), False

    # money / time
    if inp.budget is not None:
        p.budget_inr = int(round(inp.budget))
    else:
        p.budget_inr = parse_money(free) or parse_money(mood)
    mins = parse_duration_minutes(inp.available_time) if inp.available_time else None
    if mins is None and free:
        mins = parse_duration_minutes(free)
    p.minutes = mins
    p.start_min = parse_clock(inp.start_time) if inp.start_time else None
    # a clock window ("free 4-7pm") gives both the start and how long we have
    window = (parse_clock_range(inp.available_time) if inp.available_time else None) or (parse_clock_range(free) if free else None)
    if window:
        if p.start_min is None:
            p.start_min = window[0]
        if p.start_min == window[0]:
            if p.minutes is None:
                p.minutes = window[1] - window[0]
                p.assumptions.append(f"I read your window as {fmt12(window[0])} to {fmt12(window[1])}.")
        else:
            p.assumptions.append(
                f"The start time you picked ({fmt12(p.start_min)}) doesn't match the window in your note "
                f"({fmt12(window[0])} to {fmt12(window[1])}), so I used the start time."
            )
    if p.start_min is None and free and _CUE_START.search(free.lower()):
        p.start_min = parse_clock(free)
    if p.start_min is None and free:
        # daypart words are a reasonable cue on their own ("a lazy evening")
        if re.search(r"\b(morning|afternoon|evening|night)\b", free.lower()):
            p.start_min = parse_clock(free)

    # mood
    if _has(_LOW_ENERGY, text) and not _has(_HIGH_ENERGY, text):
        p.energy = "low"
    elif _has(_HIGH_ENERGY, text) and not _has(_LOW_ENERGY, text):
        p.energy = "high"
    if _has(_QUIET, text) and not _has(_LIVELY, text):
        p.social = "quiet"
    elif _has(_LIVELY, text) and not _has(_QUIET, text):
        p.social = "lively"
    p.mood_summary = mood.strip()

    # interests
    known, unknown = normalize_interests(inp.interests)
    for extra in _interests_from_text(" ".join([mood, free])):
        if extra not in known:
            known.append(extra)
    p.interests = known[:8]
    unknown = [u for u in dict.fromkeys(clean_label(x, 30) for x in unknown) if u]
    if unknown:
        p.notes.append(
            "I don't have venue data for: " + ", ".join(unknown) + " - I matched your other interests instead."
        )

    # constraints
    p.diet = _diet_from(text)
    if _EGGETARIAN.search(text):
        p.notes.append("Assumed eggs are fine for 'eggetarian' - I kept to vegetarian venues.")
    p.avoid_crowds = _has(_CROWD, text) or (p.social == "quiet" and bool(_has(re.compile(r"crowd"), text)))
    p.avoid_alcohol = _has(_ALCOHOL, text)
    p.indoor_only = _has(_INDOOR, text)
    handled = re.compile(
        r"\b(veg\w*|jain|meat\w*|crowd\w*|quiet|peaceful|alcohol\w*|drink\w*|pubs?|bars?|indoors?|"
        r"air[- ]?condition\w*|rain\w*|heat|sun|eggetarian|calm)\b"
    )
    for c in cons:
        label = clean_label(c, 80)
        if label and not handled.search(label.lower()) and label not in p.other_constraints:
            p.other_constraints.append(label)
    p.other_constraints = p.other_constraints[:5]
    return p


# ---------------------------------------------------------------- merge + defaults


def merge_llm(base: Preferences, llm: LLMPrefs, inp: UserInput) -> Preferences:
    p = base.model_copy(deep=True)
    p.parsed_by = "llm"

    if not p.city and llm.city and not p.city_ambiguous:
        city, how = match_city(llm.city)
        if city:
            p.city, p.city_display, p.city_known, p.city_input = city.key, city.display, True, llm.city
        elif how != "ambiguous" and clean_label(llm.city):
            label = clean_label(llm.city)
            p.city_input, p.city, p.city_display = label, label.lower(), label.title()
    if p.budget_inr is None and llm.budget_inr is not None and llm.budget_inr >= 0:
        p.budget_inr = int(round(llm.budget_inr))
    if p.minutes is None and llm.minutes:
        p.minutes = int(round(llm.minutes))
    if p.start_min is None and llm.start_time:
        p.start_min = parse_clock(llm.start_time)

    # the model reads mood better than keyword rules, so it decides the soft traits
    p.energy = llm.energy  # type: ignore[assignment]
    p.social = llm.social  # type: ignore[assignment]
    p.mood_summary = (llm.mood_summary or base.mood_summary).strip()[:200]

    known, unknown = normalize_interests(llm.interests)
    for k in known:
        if k not in p.interests:
            p.interests.append(k)
    p.interests = p.interests[:8]

    # hard constraints: the model may add, never remove
    if _DIET_RANK.get(llm.diet, 0) > _DIET_RANK.get(p.diet, 0):
        p.diet = llm.diet  # type: ignore[assignment]
    p.avoid_crowds = p.avoid_crowds or llm.avoid_crowds
    p.avoid_alcohol = p.avoid_alcohol or llm.avoid_alcohol
    p.indoor_only = p.indoor_only or llm.indoor_only
    for oc in llm.other_constraints:
        oc = clean_label(oc, 80)
        if oc and not re.search(r"(veg|vegan|jain|crowd|alcohol|indoor)", oc.lower()) and oc.lower() not in {
            x.lower() for x in p.other_constraints
        }:
            p.other_constraints.append(oc)
    p.other_constraints = p.other_constraints[:5]
    return p


def finalize_defaults(p: Preferences) -> Preferences:
    """Fill gaps with stated assumptions; clamp silly values."""
    if p.budget_inr is None:
        p.budget_inr = DEFAULT_BUDGET
        p.defaulted.append("budget")
        p.assumptions.append(f"No budget given, so I assumed {fmt_inr(DEFAULT_BUDGET)} for the whole outing.")
    if p.budget_inr > 1_000_000:
        p.assumptions.append(f"I capped the budget at {fmt_inr(1_000_000)}.")
    p.budget_inr = max(0, min(p.budget_inr, 1_000_000))

    if p.minutes is None:
        p.minutes = DEFAULT_MINUTES
        p.defaulted.append("time")
        p.assumptions.append("No time window given, so I assumed 4 hours.")
    if p.minutes < MIN_MINUTES:
        p.assumptions.append("That's very short - I planned for 30 minutes.")
        p.minutes = MIN_MINUTES
    if p.minutes > MAX_MINUTES:
        p.assumptions.append("I capped the outing at 12 hours.")
        p.minutes = MAX_MINUTES

    if not p.interests:
        p.interests = ["food", "walks"]
        p.defaulted.append("interests")
        p.assumptions.append("No interests given, so I assumed an easy mix of food and a walk.")
    if p.start_min is not None:
        if p.start_min > LATEST_START:
            p.assumptions.append(f"Plans can start no later than {fmt12(LATEST_START)}, so I used that.")
        p.start_min = max(0, min(p.start_min, LATEST_START))
        set_window(p, p.start_min)
    return p


def set_window(p: Preferences, start_min: int) -> None:
    """Fix the start time and derive the end (never past 23:59)."""
    p.start_min = start_min
    wanted = start_min + (p.minutes or DEFAULT_MINUTES)
    p.end_min = min(wanted, LAST_MINUTE)
    p.minutes = p.end_min - start_min
    if wanted > LAST_MINUTE and not any("midnight" in a for a in p.assumptions):
        p.assumptions.append(f"That would run past midnight, so I planned until {fmt12(LAST_MINUTE)}.")


# ---------------------------------------------------------------- tool entry point


def parse_user_preferences(inp: UserInput, llm: LLM | None, info: ToolInfo) -> Preferences:
    base = rules_parse(inp)
    needs_nuance = bool(inp.free_text or inp.mood or base.other_constraints or (inp.interests and not base.interests))
    prefs = base
    if llm is not None and needs_nuance:
        try:
            res = call_validated(llm, "parse", parse_messages(inp), LLMPrefs, effort="low", info=info)
            prefs = merge_llm(base, res, inp)
        except (LLMUnavailable, LLMBadOutput) as e:
            info.fallback = f"keyword rules ({short(e, 90)})"
            info.unavailable = isinstance(e, LLMUnavailable)
    elif llm is not None:
        info.notes.append("structured input was clear; no model call needed")
    return finalize_defaults(prefs)
