"""Curated city datasets + forgiving city-name resolution."""
from __future__ import annotations

import difflib
import json
import re
from functools import lru_cache
from pathlib import Path

from .schemas import CityData
from .util import parse_windows

DATA_DIR = Path(__file__).parent / "data"
FUZZY_CUTOFF = 0.85  # 0.8 let "Bangor" through as Bangalore


def _check(city: CityData) -> None:
    """Fail fast (at import/startup and in tests) if a dataset is internally inconsistent."""
    seen: set[str] = set()
    for p in city.places:
        if p.id in seen:
            raise ValueError(f"{city.key}: duplicate place id {p.id}")
        seen.add(p.id)
        if p.area not in city.areas and (p.lat is None or p.lon is None):
            raise ValueError(f"{city.key}: place {p.id} has unknown area '{p.area}' and no coordinates")
        if not parse_windows(p.hours):
            raise ValueError(f"{city.key}: place {p.id} has unparseable hours '{p.hours}'")
        for w in p.peak:
            parse_windows(w)


@lru_cache(maxsize=1)
def load_cities() -> dict[str, CityData]:
    out: dict[str, CityData] = {}
    for f in sorted(DATA_DIR.glob("*.json")):
        city = CityData.model_validate(json.loads(f.read_text(encoding="utf-8")))
        _check(city)
        out[city.key] = city
    return out


def get_city(key: str | None) -> CityData | None:
    return load_cities().get(key or "")


def supported_cities() -> list[dict]:
    return [{"key": c.key, "display": c.display, "places": len(c.places)} for c in load_cities().values()]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())).strip()


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for c in load_cities().values():
        for a in [c.key, c.display, *c.aliases]:
            idx[_norm(a)] = c.key
    return idx


def _mentions(n: str) -> list[CityData]:
    """Distinct supported cities named as whole words in normalised text, in order of appearance."""
    found: dict[str, int] = {}
    for alias, key in _alias_index().items():
        m = re.search(rf"\b{re.escape(alias)}\b", n)
        if m and (key not in found or m.start() < found[key]):
            found[key] = m.start()
    cities = load_cities()
    return [cities[k] for k, _ in sorted(found.items(), key=lambda kv: kv[1])]


def match_city(text: str | None, *, fuzzy: bool = True) -> tuple[CityData | None, str]:
    """(city, how) with how in exact | contains | fuzzy | ambiguous | none.

    'ambiguous' means several supported cities were named ('Pune or Bangalore'): we return no city
    so the agent can ask, rather than silently picking one.
    """
    n = _norm(text or "")
    if not n:
        return None, "none"
    idx = _alias_index()
    if n in idx:
        return load_cities()[idx[n]], "exact"
    found = _mentions(n)
    if len(found) == 1:
        return found[0], "contains"
    if len(found) > 1:
        return None, "ambiguous"
    if fuzzy:
        close = difflib.get_close_matches(n, list(idx), n=1, cutoff=FUZZY_CUTOFF)
        if close:
            return load_cities()[idx[close[0]]], "fuzzy"
    return None, "none"


def resolve_city(text: str | None) -> CityData | None:
    """'Banglore', 'Bengaluru, India', 'gurgaon' -> the matching dataset; None if unknown or ambiguous."""
    return match_city(text)[0]


def find_city_in_text(text: str | None) -> CityData | None:
    """Spot a supported city name inside free text ('plan a saturday in Pune'). No guessing from typos."""
    return match_city(text, fuzzy=False)[0]
