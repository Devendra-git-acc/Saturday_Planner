"""Domain rules, travel model, city resolution and the integrity of the three curated datasets."""
from __future__ import annotations

import pytest

from app import geo
from app.cities import find_city_in_text, get_city, load_cities, match_city, resolve_city, supported_cities
from app.rules import (
    crowd_at, diet_check, energy_gap, meal_ok, meals_at, noise_gap, normalize_interests, window_covers_meal,
)
from app.schemas import INTEREST_VOCAB, Place
from app.util import parse_windows


def place(**kw) -> Place:
    base = dict(id="x", name="X", kind="food", area="a", diet="unknown")
    base.update(kw)
    return Place(**base)


# ------------------------------------------------------------------ interests


def test_normalize_interests_maps_synonyms_and_reports_unknowns():
    known, unknown = normalize_interests(["Live Music", "foodie", "walking", "underwater basket weaving", "", "  "])
    assert known == ["music", "food", "walks"]
    assert unknown == ["underwater basket weaving"]


def test_normalize_interests_dedupes_and_finds_a_vocabulary_word_inside_a_phrase():
    known, _ = normalize_interests(["food", "Food", "street food walks", "FOOD "])
    assert known == ["food"]


# ---------------------------------------------------------------------- diet


@pytest.mark.parametrize("diet,place_kw,allowed,caution_contains", [
    ("vegetarian", dict(diet="pure_veg"), True, None),
    ("vegetarian", dict(diet="veg_friendly"), True, None),
    ("vegetarian", dict(diet="non_veg_focus"), False, None),
    ("vegetarian", dict(diet="unknown"), True, "vegetarian"),
    ("vegan", dict(diet="veg_friendly", vegan_friendly=True), True, None),
    ("vegan", dict(diet="pure_veg"), True, "vegan"),
    ("vegan", dict(diet="pure_veg", dairy_centric=True), False, None),
    ("vegan", dict(diet="veg_friendly"), False, None),
    ("vegan", dict(diet="non_veg_focus"), False, None),
    ("jain", dict(diet="pure_veg", jain_friendly=True), True, None),
    ("jain", dict(diet="pure_veg"), True, "Jain"),
    ("jain", dict(diet="veg_friendly"), False, None),
    ("none", dict(diet="non_veg_focus"), True, None),
])
def test_diet_check(diet, place_kw, allowed, caution_contains):
    ok, caution = diet_check(place(**place_kw), diet)
    assert ok is allowed
    if caution_contains:
        assert caution_contains in (caution or "")
    else:
        assert caution is None


def test_activities_are_never_diet_filtered():
    assert diet_check(place(kind="activity", diet="na"), "vegan") == (True, None)


# ------------------------------------------------------------ crowd / energy


def test_crowd_peak_windows_bump_one_level():
    p = place(crowd="medium", peak=["17:00-20:00"])
    assert crowd_at(p, 16 * 60) == "medium"
    assert crowd_at(p, 17 * 60) == "high"
    assert crowd_at(p, 20 * 60) == "medium"  # window end is exclusive
    assert crowd_at(place(crowd="low", peak=["10:00-12:00"]), 11 * 60) == "medium"
    assert crowd_at(place(crowd="high", peak=["10:00-12:00"]), 11 * 60) == "high"


@pytest.mark.parametrize("user,place_energy,gap", [("low", "low", 0), ("low", "medium", 1), ("low", "high", 2), ("high", "low", 0), ("medium", "high", 1)])
def test_energy_gap(user, place_energy, gap):
    assert energy_gap(user, place_energy) == gap


@pytest.mark.parametrize("social,noise,gap", [("quiet", "quiet", 0), ("quiet", "moderate", 1), ("quiet", "loud", 2), ("balanced", "loud", 1), ("balanced", "moderate", 0), ("lively", "loud", 0)])
def test_noise_gap(social, noise, gap):
    assert noise_gap(social, noise) == gap


# --------------------------------------------------------------------- meals


def test_meal_windows_with_tolerance():
    assert "lunch" in meals_at(13 * 60)
    assert "breakfast" in meals_at(10 * 60)
    assert "dinner" in meals_at(20 * 60)
    assert {"lunch", "snack"} <= meals_at(15 * 60 + 45)  # the windows touch, tolerance overlaps them
    assert meals_at(3 * 60) == set()


def test_meal_ok_depends_on_what_the_place_serves():
    lunch_place = place(meal=["lunch"])
    assert meal_ok(lunch_place, 12 * 60 + 30, 13 * 60 + 30)
    assert not meal_ok(lunch_place, 20 * 60, 21 * 60)
    assert meal_ok(place(kind="activity", diet="na"), 3 * 60, 4 * 60)  # not a food stop
    assert meal_ok(place(meal=[]), 3 * 60, 4 * 60)  # no meal info: nothing to object to


def test_window_covers_meal_needs_an_hour_of_overlap():
    assert window_covers_meal(10 * 60, 14 * 60) == ["breakfast", "lunch"]
    assert window_covers_meal(13 * 60, 13 * 60 + 40) == []
    assert "dinner" in window_covers_meal(18 * 60, 22 * 60)


# ----------------------------------------------------------------------- geo


@pytest.fixture(scope="module")
def blr():
    return get_city("bangalore")


def _pair(city, a_id, b_id):
    by = city.by_id()
    return by[a_id], by[b_id]


def test_leg_start_and_stay(blr):
    a, _ = _pair(blr, "blr-cubbon", "blr-adigas")
    assert geo.leg(blr, None, a, 600).mode == "start"
    stay = geo.leg(blr, a, a, 600)
    assert stay.mode == "stay" and stay.minutes == 0 and stay.cost == 0


def test_far_hops_are_rides_with_a_fare_and_near_hops_are_walks(blr):
    cubbon, ime = _pair(blr, "blr-cubbon", "blr-ime")
    ride = geo.leg(blr, cubbon, ime, 12 * 60)
    assert ride.mode == "auto/cab" and ride.cost >= 60 and ride.cost % 10 == 0 and ride.minutes > 10
    walks = [
        geo.leg(blr, a, b, 600) for a in blr.places for b in blr.places
        if a.id != b.id and geo.distance_km(blr, a, b) <= geo.WALK_MAX_KM
    ]
    assert walks, "the dataset should contain at least one walkable pair"
    assert all(w.mode == "walk" and w.cost == 0 and w.minutes >= 4 for w in walks)


def test_distance_is_symmetric_and_evenings_are_slower(blr):
    a, b = _pair(blr, "blr-cubbon", "blr-ime")
    assert geo.distance_km(blr, a, b) == geo.distance_km(blr, b, a)
    morning, evening = geo.leg(blr, a, b, 8 * 60), geo.leg(blr, a, b, 18 * 60)
    assert evening.minutes > morning.minutes
    assert geo.traffic_factor(18 * 60) < geo.traffic_factor(13 * 60) < geo.traffic_factor(8 * 60)


def test_fare_has_a_floor_and_grows_with_distance():
    assert geo.fare(0) == 60
    assert geo.fare(10) == 200
    assert geo.fare(20) > geo.fare(10)


# -------------------------------------------------------------------- cities


@pytest.mark.parametrize("text,key", [
    ("Bangalore", "bangalore"), ("  BANGALORE  ", "bangalore"), ("Bengaluru, Karnataka", "bangalore"), ("Banglore", "bangalore"),
    ("blr", "bangalore"), ("Gurgaon", "gurugram"), ("Gurgoan", "gurugram"), ("gurugram haryana", "gurugram"),
    ("Pune, India", "pune"), ("Poona", "pune"), ("Puna", "pune"), ("bangalore!!!", "bangalore"),
])
def test_resolve_city_is_forgiving(text, key):
    assert resolve_city(text).key == key


@pytest.mark.parametrize("text", [None, "", "   ", "Mumbai", "Delhi", "Bangor", "Bengal", "Atlantis", "!!!"])
def test_resolve_city_returns_none_for_unknown_places(text):
    assert resolve_city(text) is None


def test_typo_corrections_are_flagged_as_fuzzy_and_exact_matches_are_not():
    assert match_city("Banglore")[1] == "fuzzy"
    assert match_city("Bangalore")[1] == "exact"
    assert match_city("Bengaluru, Karnataka")[1] == "contains"


@pytest.mark.parametrize("text", ["Pune or Bangalore", "pune/bangalore", "Gurgaon and Pune"])
def test_naming_two_cities_is_ambiguous_not_a_silent_pick(text):
    city, how = match_city(text)
    assert city is None and how == "ambiguous"
    assert find_city_in_text(text) is None


def test_find_city_in_text():
    assert find_city_in_text("plan a saturday in Pune").key == "pune"
    assert find_city_in_text("gurgaon trip please").key == "gurugram"
    assert find_city_in_text("something fun") is None
    assert find_city_in_text("I live near Bengaluru airport").key == "bangalore"
    assert find_city_in_text("Banglore") is None  # free text never guesses from typos


def test_supported_cities_listing():
    listing = {c["key"]: c for c in supported_cities()}
    assert set(listing) == {"bangalore", "pune", "gurugram"}
    assert all(c["places"] >= 14 for c in listing.values())


# ---------------------------------------------------------- dataset integrity


BOUNDS = {  # generous boxes around each city: catches a swapped lat/lon or a place filed under the wrong city
    "bangalore": (12.7, 13.3, 77.3, 77.9),
    "pune": (18.3, 18.8, 73.6, 74.1),
    "gurugram": (28.3, 28.6, 76.8, 77.2),
}


@pytest.mark.parametrize("key", ["bangalore", "pune", "gurugram"])
class TestDatasets:
    def test_basic_shape(self, key):
        c = load_cities()[key]
        kinds = {p.kind for p in c.places}
        assert kinds == {"activity", "food"}
        assert sum(p.kind == "activity" for p in c.places) >= 9 and sum(p.kind == "food" for p in c.places) >= 5
        assert len({p.id for p in c.places}) == len(c.places)
        assert c.avg_speed_kmph > 5

    def test_areas_are_inside_the_city(self, key):
        c = load_cities()[key]
        lat0, lat1, lon0, lon1 = BOUNDS[key]
        for name, a in c.areas.items():
            assert lat0 <= a.lat <= lat1 and lon0 <= a.lon <= lon1, f"{key}/{name} is outside the {key} box"

    def test_every_place_is_well_formed(self, key):
        c = load_cities()[key]
        for p in c.places:
            assert p.area in c.areas, p.id
            assert p.cost >= 0 and p.duration > 0, p.id
            assert p.min_dur <= p.duration <= p.max_dur, p.id
            assert parse_windows(p.hours), p.id
            assert p.tags and set(p.tags) <= set(INTEREST_VOCAB), (p.id, p.tags)
            assert p.name.strip() and p.note.strip(), p.id
            if p.kind == "food":
                assert p.diet in {"pure_veg", "veg_friendly", "non_veg_focus", "unknown"}, p.id
                assert p.meal, f"{p.id}: a food venue needs the meals it serves"
            else:
                assert p.diet == "na", p.id

    def test_hours_end_after_they_start(self, key):
        for p in load_cities()[key].places:
            for s, e in parse_windows(p.hours):
                assert e > s, p.id

    def test_a_vegetarian_can_eat_somewhere(self, key):
        foods = [p for p in load_cities()[key].places if p.kind == "food"]
        assert sum(diet_check(p, "vegetarian")[0] for p in foods) >= 3
        assert sum(diet_check(p, "vegan")[0] for p in foods) >= 1
