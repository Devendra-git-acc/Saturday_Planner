"""Bonus tool - lookupLiveVenues: build a rough city dataset from OpenStreetMap on demand.

Used only for cities without a curated dataset. OSM gives names, coordinates and (sometimes)
opening hours and diet tags; prices, crowds and durations are category-level ESTIMATES and every
venue is marked as such. Be a good citizen: identify the app, throttle Nominatim, cache results.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

import httpx

from ..schemas import Area, CityData, Place
from ..util import short

log = logging.getLogger("saturday.osm")

NOMINATIM = "https://nominatim.openstreetmap.org/search"
# Public Overpass servers are volunteer-run and often overloaded (502/504), so try more than one.
OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
OVERPASS = OVERPASS_URLS[0]
RADIUS_M = 4000  # a dense megacity with a wide radius makes the public servers time out
LOOKUP_BUDGET_S = 25.0  # all Overpass attempts together
PER_SERVER_S = 9.0
CACHE_TTL_S = 24 * 3600
NEG_TTL_S = 10 * 60
MAX_ACTIVITIES, MAX_FOOD = 30, 24

_cache: dict[str, tuple[float, CityData | None]] = {}
_lock = threading.Lock()  # guards the cache only
_throttle_lock = threading.Lock()  # serialises calls to Nominatim (about one per second)
_last_nominatim = 0.0

# (kind, category, tags, cost, duration, hours, energy, noise, meal)
_CATEGORIES: dict[tuple[str, str], tuple] = {
    ("leisure", "park"): ("activity", "park", ["nature", "walks"], 0, 60, "06:00-19:00", "low", "quiet", []),
    ("leisure", "garden"): ("activity", "garden", ["nature", "walks", "photography"], 0, 60, "06:00-19:00", "low", "quiet", []),
    ("leisure", "nature_reserve"): ("activity", "nature reserve", ["nature", "walks"], 50, 90, "07:00-17:00", "medium", "quiet", []),
    ("tourism", "museum"): ("activity", "museum", ["history", "art"], 100, 75, "10:00-17:30", "low", "quiet", []),
    ("tourism", "gallery"): ("activity", "gallery", ["art"], 0, 50, "11:00-18:00", "low", "quiet", []),
    ("tourism", "attraction"): ("activity", "attraction", ["photography"], 50, 60, "09:00-18:00", "low", "moderate", []),
    ("tourism", "viewpoint"): ("activity", "viewpoint", ["nature", "photography"], 0, 30, "06:00-19:00", "low", "quiet", []),
    ("tourism", "zoo"): ("activity", "zoo", ["nature"], 150, 120, "09:30-17:30", "medium", "moderate", []),
    ("historic", "*"): ("activity", "historic site", ["history", "photography"], 25, 50, "09:00-17:30", "low", "quiet", []),
    ("amenity", "theatre"): ("activity", "theatre", ["theatre", "music", "art"], 400, 120, "16:00-22:00", "low", "quiet", []),
    ("amenity", "arts_centre"): ("activity", "arts centre", ["art", "music"], 150, 75, "11:00-20:00", "low", "quiet", []),
    ("amenity", "library"): ("activity", "library", ["books"], 0, 60, "10:00-18:00", "low", "quiet", []),
    ("amenity", "cafe"): ("food", "cafe", ["coffee", "food"], 300, 45, "08:00-22:00", "low", "moderate", ["breakfast", "snack"]),
    ("amenity", "restaurant"): ("food", "restaurant", ["food"], 450, 60, "11:30-22:30", "low", "moderate", ["lunch", "dinner"]),
    ("amenity", "ice_cream"): ("food", "dessert", ["food"], 150, 25, "11:00-22:00", "low", "moderate", ["snack"]),
}

_QUERY = """[out:json][timeout:15];
(
  nwr(around:{r},{lat},{lon})["leisure"~"^(park|garden|nature_reserve)$"]["name"];
  nwr(around:{r},{lat},{lon})["tourism"~"^(museum|gallery|attraction|viewpoint|zoo)$"]["name"];
  nwr(around:{r},{lat},{lon})["historic"~"^(monument|fort|castle|memorial|archaeological_site)$"]["name"];
  nwr(around:{r},{lat},{lon})["amenity"~"^(theatre|arts_centre|library|cafe|restaurant|ice_cream)$"]["name"];
);
out center 120;"""

_DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]


def _clean(name: str) -> str:
    return re.sub(r"[\x00-\x1f<>]", "", name).strip()[:80]


def _window(a: str, b: str) -> str | None:
    """'9:00','17:30' -> '09:00-17:30'. Hours past midnight run to the end of our day; nonsense is dropped."""
    ma, mb = re.fullmatch(r"(\d{1,2}):(\d{2})", a), re.fullmatch(r"(\d{1,2}):(\d{2})", b)
    if not ma or not mb:
        return None
    h1, m1, h2, m2 = int(ma[1]), int(ma[2]), int(mb[1]), int(mb[2])
    if h1 > 23 or m1 > 59 or h2 > 24 or m2 > 59:
        return None
    start, end = h1 * 60 + m1, h2 * 60 + m2
    if end <= start or end >= 24 * 60:
        end = 24 * 60 - 1  # open past midnight (or 24:00): open until the end of our day
    return None if end - start < 30 else f"{start // 60:02d}:{start % 60:02d}-{end // 60:02d}:{end % 60:02d}"


def parse_saturday_hours(spec: str | None) -> str | None | bool:
    """Return 'HH:MM-HH:MM[,..]' for Saturday, False if closed that day, None if unknown/unparseable.

    A weekly schedule that never mentions Saturday ("Mo-Fr 09:00-17:00") means closed on Saturday.
    Anything we cannot read (seasons, "sunrise-sunset", "by appointment") stays unknown, not closed.
    """
    if not spec:
        return None
    spec = spec.strip()
    if spec == "24/7":
        return "00:00-23:59"
    found: str | None = None
    saw_weekly_rule = covered_any = False
    for rule in (r.strip() for r in spec.split(";")):
        m = re.match(r"^([A-Za-z,\- ]+?)\s+(.+)$", rule)
        if not m:
            continue
        days, times = m.group(1).replace(" ", ""), m.group(2).strip()
        covered = valid = False
        for part in days.split(","):
            if "-" in part:
                a, _, b = part.partition("-")
                if a in _DAYS and b in _DAYS:
                    valid = True
                    i, j = _DAYS.index(a), _DAYS.index(b)
                    rng = range(i, j + 1) if i <= j else [*range(i, 7), *range(0, j + 1)]
                    covered = covered or 5 in rng
            elif part in _DAYS:
                valid = True
                covered = covered or part == "Sa"
        saw_weekly_rule = saw_weekly_rule or valid
        if not covered:
            continue
        covered_any = True
        if times.lower() in ("off", "closed"):
            return False
        wins = [w for a, b in re.findall(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", times) if (w := _window(a, b))]
        if wins:
            found = ",".join(wins)
    if found is None and saw_weekly_rule and not covered_any:
        return False
    return found


def _to_place(el: dict[str, Any], area_key: str) -> Place | None:
    tags = el.get("tags") or {}
    name = _clean(str(tags.get("name", "")))
    if not name:
        return None
    lat = el.get("lat", (el.get("center") or {}).get("lat"))
    lon = el.get("lon", (el.get("center") or {}).get("lon"))
    if lat is None or lon is None:
        return None
    cat = None
    for key in ("leisure", "tourism", "amenity"):
        if tags.get(key) and (key, tags[key]) in _CATEGORIES:
            cat = _CATEGORIES[(key, tags[key])]
            break
    if cat is None and tags.get("historic"):
        cat = _CATEGORIES[("historic", "*")]
    if cat is None:
        return None
    kind, category, ptags, cost, dur, hours, energy, noise, meal = cat

    hrs = parse_saturday_hours(tags.get("opening_hours"))
    if hrs is False:
        return None  # closed on Saturdays
    verified_hours = isinstance(hrs, str)

    diet = "unknown" if kind == "food" else "na"
    if kind == "food":
        veg = str(tags.get("diet:vegetarian", "")).lower()
        diet = "pure_veg" if veg == "only" else "veg_friendly" if veg == "yes" else "unknown"
    vegan = str(tags.get("diet:vegan", "")).lower() in ("yes", "only")
    note = "Listed on OpenStreetMap."
    if tags.get("cuisine"):
        note = f"Cuisine: {str(tags['cuisine']).replace(';', ', ')[:60]}."
    return Place(
        id=f"osm-{el.get('type', 'x')[0]}{el.get('id')}",
        name=name, kind=kind, category=category, area=area_key,  # type: ignore[arg-type]
        tags=list(ptags), cost=cost, duration=dur, hours=hrs if verified_hours else hours,  # type: ignore[arg-type]
        energy=energy, crowd="medium", noise=noise, indoor=kind == "food" or category in ("museum", "gallery", "theatre", "library"),
        diet=diet, vegan_friendly=vegan, meal=list(meal), note=note,
        verify="Price, hours and crowd level are estimates" + ("" if not verified_hours else " (hours from OpenStreetMap)") + " - check before going.",
        lat=float(lat), lon=float(lon), source="openstreetmap",
    )


def _throttle() -> None:
    global _last_nominatim
    with _throttle_lock:
        wait = 1.1 - (time.monotonic() - _last_nominatim)
        if wait > 0:
            time.sleep(wait)
        _last_nominatim = time.monotonic()


def _overpass(http: httpx.Client, query: str, ua: str) -> list[dict[str, Any]]:
    """POST the query to each public server in turn within one time budget; the first good answer wins."""
    last: Exception | None = None
    t0 = time.monotonic()
    for url in OVERPASS_URLS:
        left = LOOKUP_BUDGET_S - (time.monotonic() - t0)
        if left < 4:
            break
        try:
            r = http.post(url, data={"data": query}, headers={"User-Agent": ua}, timeout=min(PER_SERVER_S, left))
            r.raise_for_status()
            return r.json().get("elements", [])
        except (httpx.HTTPError, ValueError) as e:  # 5xx, 429, timeouts, connection errors, non-JSON bodies
            log.warning("Overpass server %s failed: %s", url.split("/")[2], short(e))
            last = e
    raise last or RuntimeError("no Overpass server answered in time")


def fetch_city(name: str, *, contact: str = "", client: httpx.Client | None = None, timeout: float = 20.0) -> CityData | None:
    """Look up a city and build a CityData from nearby OSM venues. None on any failure."""
    key = re.sub(r"\s+", " ", name.strip().lower())
    if not key:
        return None
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now < hit[0]:
            return hit[1]

    ua = f"perfect-saturday-planner/1.0 ({contact or 'demo app'})"
    own = client is None
    http = client or httpx.Client(timeout=timeout, headers={"User-Agent": ua})
    result: CityData | None = None
    try:
        _throttle()
        r = http.get(NOMINATIM, params={"q": name, "format": "jsonv2", "limit": 1}, headers={"User-Agent": ua})
        r.raise_for_status()
        hits = r.json()
        if not hits:
            raise LookupError("city not found on OpenStreetMap")
        lat, lon = float(hits[0]["lat"]), float(hits[0]["lon"])
        display = name.strip().title()

        q = _QUERY.format(r=RADIUS_M, lat=lat, lon=lon)
        elements = _overpass(http, q, ua)
        places: list[Place] = []
        seen: set[str] = set()
        elements.sort(key=lambda e: -len(e.get("tags") or {}))  # better-documented venues first
        counts = {"activity": 0, "food": 0}
        for el in elements:
            p = _to_place(el, "city")
            if p is None or p.name.lower() in seen:
                continue
            cap = MAX_ACTIVITIES if p.kind == "activity" else MAX_FOOD
            if counts[p.kind] >= cap:
                continue
            seen.add(p.name.lower())
            counts[p.kind] += 1
            places.append(p)
        if len(places) >= 3:
            result = CityData(
                key=key, display=display, aliases=[], avg_speed_kmph=18.0,
                areas={"city": Area(name=f"{display} (centre)", lat=lat, lon=lon)}, places=places, source="openstreetmap",
            )
    except Exception as e:  # noqa: BLE001 - any failure means 'no live data', never a crash
        log.warning("OSM lookup failed for %r: %s", name, short(e))
    finally:
        if own:
            http.close()

    with _lock:
        _cache[key] = (time.monotonic() + (CACHE_TTL_S if result else NEG_TTL_S), result)
    return result


def clear_cache() -> None:
    with _lock:
        _cache.clear()