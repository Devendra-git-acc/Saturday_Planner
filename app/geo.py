"""Travel model: distance -> minutes and rupees.

Deliberately simple and honest: neighbourhood centroids, haversine x road factor, a city-specific
average speed that drops in the evening, walking for short hops, and a rough auto/cab fare.
"""
from __future__ import annotations

import math

from .schemas import CityData, Leg, Place
from .util import haversine_km

ROAD_FACTOR = 1.35  # straight-line -> road distance
WALK_KMPH = 4.5
WALK_MAX_KM = 1.2
SAME_AREA_KM = 0.6  # two places in one neighbourhood are rarely on top of each other
MIN_DIFF_AREA_KM = 1.2
HAIL_BUFFER_MIN = 6  # finding an auto/cab, parking, walking to the door


def coords(city: CityData, p: Place) -> tuple[float, float]:
    if p.lat is not None and p.lon is not None:
        return p.lat, p.lon
    a = city.areas[p.area]
    return a.lat, a.lon


def traffic_factor(minute: int) -> float:
    """Multiplier on average speed by departure time (Saturday)."""
    if 17 * 60 <= minute < 21 * 60 + 30:
        return 0.72
    if 12 * 60 <= minute < 17 * 60:
        return 0.9
    if minute < 9 * 60 + 30:
        return 1.1
    return 1.0


def fare(km: float) -> int:
    """Rough auto/cab fare in INR, rounded to ten."""
    return int(round(max(60.0, 45 + 15 * km) / 10.0)) * 10


def distance_km(city: CityData, a: Place, b: Place) -> float:
    key = (a.id, b.id) if a.id <= b.id else (b.id, a.id)
    cached = city._km_cache.get(key)
    if cached is not None:
        return cached
    (la, lo), (lb, lob) = coords(city, a), coords(city, b)
    km = haversine_km(la, lo, lb, lob) * ROAD_FACTOR
    km = max(km, SAME_AREA_KM if a.area == b.area else MIN_DIFF_AREA_KM)
    city._km_cache[key] = km
    return km


def leg(city: CityData, a: Place | None, b: Place, depart_min: int) -> Leg:
    """Travel from stop `a` (None = start of day) to `b`, leaving at `depart_min`."""
    if a is None:
        return Leg(minutes=0, cost=0, mode="start", km=0.0)
    if a.id == b.id:
        return Leg(minutes=0, cost=0, mode="stay", km=0.0)
    km = distance_km(city, a, b)
    if km <= WALK_MAX_KM:
        return Leg(minutes=max(4, math.ceil(km / WALK_KMPH * 60) + 1), cost=0, mode="walk", km=round(km, 1))
    speed = max(6.0, city.avg_speed_kmph * traffic_factor(depart_min))
    minutes = math.ceil(km / speed * 60) + HAIL_BUFFER_MIN
    return Leg(minutes=minutes, cost=fare(km), mode="auto/cab", km=round(km, 1))


def area_minutes(city: CityData, a: str, b: str, depart_min: int = 12 * 60) -> int:
    """Approximate door-to-door minutes between two neighbourhoods (used as a hint for the LLM)."""
    if a == b:
        return 8
    A, B = city.areas.get(a), city.areas.get(b)
    if A is None or B is None:
        return 30
    km = max(haversine_km(A.lat, A.lon, B.lat, B.lon) * ROAD_FACTOR, MIN_DIFF_AREA_KM)
    if km <= WALK_MAX_KM:
        return max(4, math.ceil(km / WALK_KMPH * 60) + 1)
    speed = max(6.0, city.avg_speed_kmph * traffic_factor(depart_min))
    return math.ceil(km / speed * 60) + HAIL_BUFFER_MIN


def area_matrix(city: CityData, keys: list[str], depart_min: int = 12 * 60) -> dict[str, int]:
    """'a|b' -> minutes, for every unordered pair of the given areas."""
    ks = sorted(set(keys))
    return {f"{a}|{b}": area_minutes(city, a, b, depart_min) for i, a in enumerate(ks) for b in ks[i + 1 :]}
