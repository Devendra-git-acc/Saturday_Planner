"""Domain rules shared by the option filter, the planner and the validator.

Keeping these in one place means the same definition of 'vegetarian-safe', 'crowded' or 'lunch
time' is used when we choose candidates and when we check the final plan.
"""
from __future__ import annotations

import re

from .schemas import INTEREST_VOCAB, Place
from .util import parse_windows

LEVELS = {"low": 0, "medium": 1, "high": 2}
LEVEL_NAMES = ("low", "medium", "high")

# Saturday meal windows (minutes after midnight). Tolerance is added in `meals_at`.
MEAL_WINDOWS = {
    "breakfast": (7 * 60, 11 * 60 + 30),
    "lunch": (12 * 60, 15 * 60 + 30),
    "snack": (15 * 60 + 30, 18 * 60 + 45),
    "dinner": (19 * 60, 22 * 60 + 45),
}
MEAL_TOLERANCE = 30

# --------------------------------------------------------------- interests

INTEREST_SYNONYMS: dict[str, str] = {
    "foodie": "food", "eating": "food", "street food": "food", "restaurants": "food", "dining": "food",
    "cuisine": "food", "snacks": "food", "dessert": "food", "sweets": "food", "tiffin": "food",
    "cafe": "coffee", "cafes": "coffee", "chai": "coffee", "tea": "coffee", "brunch": "coffee",
    "live music": "music", "concert": "music", "concerts": "music", "gigs": "music", "songs": "music",
    "singing": "music", "jazz": "music", "classical music": "music", "band": "music", "bands": "music",
    "walk": "walks", "walking": "walks", "stroll": "walks", "strolling": "walks", "hiking": "walks",
    "trek": "walks", "trekking": "walks", "jogging": "walks",
    "park": "nature", "parks": "nature", "garden": "nature", "gardens": "nature", "lake": "nature",
    "lakes": "nature", "outdoors": "nature", "greenery": "nature", "birds": "nature",
    "birdwatching": "nature", "bird watching": "nature",
    "gallery": "art", "galleries": "art", "museum": "art", "museums": "art", "painting": "art",
    "heritage": "history", "fort": "history", "forts": "history", "palace": "history",
    "monuments": "history", "architecture": "history", "culture": "history",
    "reading": "books", "bookstore": "books", "bookstores": "books", "bookshop": "books",
    "library": "books", "book": "books",
    "market": "markets", "flea market": "markets", "bazaar": "markets", "bazaars": "markets",
    "mall": "shopping", "malls": "shopping", "shop": "shopping", "browsing": "shopping",
    "pubs": "nightlife", "pub": "nightlife", "bars": "nightlife", "clubbing": "nightlife",
    "party": "nightlife", "drinks": "nightlife",
    "plays": "theatre", "drama": "theatre", "theater": "theatre", "stage": "theatre",
    "movie": "movies", "cinema": "movies", "film": "movies", "films": "movies",
    "photos": "photography", "camera": "photography", "photo walk": "photography",
    "spa": "wellness", "yoga": "wellness", "meditation": "wellness", "relaxing": "wellness",
    "temple": "spirituality", "temples": "spirituality", "religious": "spirituality",
    "board games": "games", "gaming": "games", "arcade": "games", "bowling": "games",
    "museum of science": "science", "science museum": "science", "planetarium": "science",
}


def normalize_interests(items: list[str]) -> tuple[list[str], list[str]]:
    """Map free-form interests onto our vocabulary. Returns (known, unknown) - order preserved."""
    known: list[str] = []
    unknown: list[str] = []
    for raw in items or []:
        s = re.sub(r"\s+", " ", str(raw).strip().lower())
        if not s:
            continue
        hit = s if s in INTEREST_VOCAB else INTEREST_SYNONYMS.get(s)
        if hit is None:
            for token in re.findall(r"[a-z]+", s):
                if token in INTEREST_VOCAB:
                    hit = token
                    break
                if token in INTEREST_SYNONYMS:
                    hit = INTEREST_SYNONYMS[token]
                    break
        if hit:
            if hit not in known:
                known.append(hit)
        elif s not in unknown:
            unknown.append(s)
    return known, unknown


# -------------------------------------------------------------------- diet


def diet_check(place: Place, diet: str) -> tuple[bool, str | None]:
    """(allowed, caution). Only food venues are diet-checked."""
    if place.kind != "food" or diet in ("none", "", None):
        return True, None
    d = place.diet
    if diet == "vegetarian":
        if d == "non_veg_focus":
            return False, None
        if d == "unknown":
            return True, "Confirm they have vegetarian options"
        return True, None
    if diet == "vegan":
        if place.vegan_friendly:
            return True, None
        if d == "pure_veg" and not place.dairy_centric:
            return True, "Ask for vegan preparation (no ghee, butter, paneer or curd)"
        return False, None
    if diet == "jain":
        if place.jain_friendly:
            return True, None
        if d == "pure_veg":
            return True, "Ask for Jain preparation (no onion, garlic or root vegetables)"
        return False, None
    return True, None


# ------------------------------------------------------------------ crowds


def crowd_at(place: Place, minute: int) -> str:
    """Expected Saturday crowd level at a given time: baseline, +1 level inside a peak window."""
    lvl = LEVELS.get(place.crowd, 1)
    for p in place.peak:
        for s, e in parse_windows(p):
            if s <= minute < e:
                lvl = min(2, lvl + 1)
                break
    return LEVEL_NAMES[lvl]


# -------------------------------------------------------------------- fit


def energy_gap(user_energy: str, place_energy: str) -> int:
    """How much more effort the place needs than the user has. 0 = fine."""
    return max(0, LEVELS.get(place_energy, 0) - LEVELS.get(user_energy, 1))


def noise_gap(social: str, noise: str) -> int:
    rank = {"quiet": 0, "moderate": 1, "loud": 2}
    n = rank.get(noise, 1)
    if social == "quiet":
        return n
    if social == "balanced":
        return 1 if n == 2 else 0
    return 0


# ------------------------------------------------------------------- meals


def meals_at(minute: int) -> set[str]:
    return {
        m for m, (s, e) in MEAL_WINDOWS.items() if s - MEAL_TOLERANCE <= minute < e + MEAL_TOLERANCE
    }


def meal_ok(place: Place, start: int, end: int) -> bool:
    """Is a food stop at a sensible time for what the place serves?"""
    if place.kind != "food" or not place.meal:
        return True
    mid = (start + end) // 2
    return bool(set(place.meal) & meals_at(mid))


def window_covers_meal(start: int, end: int) -> list[str]:
    """Meals whose window overlaps the plan window by at least 60 minutes."""
    out = []
    for m, (s, e) in MEAL_WINDOWS.items():
        if min(e, end) - max(s, start) >= 60:
            out.append(m)
    return out
