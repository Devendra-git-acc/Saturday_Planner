"""The OpenStreetMap bonus tool, against recorded-style fake responses.

These tests cannot prove the live Nominatim and Overpass services behave this way (the build sandbox
blocks them); they prove that OUR code handles the documented response shapes and every failure.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time

import httpx
import pytest

from app.agent import Agent
from app.rules import diet_check
from app.schemas import UserInput
from app.tools import osm
from app.trace import Tracer
from app.util import parse_windows

CENTRE = {"lat": "19.0760", "lon": "72.8777"}


def el(kind="node", id_=1, name="Place", lat=19.07, lon=72.87, **tags):
    base = {"type": kind, "id": id_, "tags": {"name": name, **tags}}
    if kind == "node":
        base.update(lat=lat, lon=lon)
    else:
        base["center"] = {"lat": lat, "lon": lon}
    return base


def good_elements() -> list[dict]:
    return [
        el("way", 10, "Joggers Park", leisure="park"),
        el("node", 11, "City Museum", lat=19.08, lon=72.88, tourism="museum", opening_hours="Mo-Su 10:00-18:00", website="x", wheelchair="yes"),
        el("node", 12, "Green Leaf Cafe", lat=19.075, lon=72.875, amenity="cafe", **{"diet:vegetarian": "only", "diet:vegan": "yes"}),
        el("node", 13, "Spice Kitchen", amenity="restaurant", cuisine="south_indian;vegetarian", **{"diet:vegetarian": "yes"}),
        el("relation", 14, "Old Fort", historic="fort"),
        el("node", 15, "Closed Saturdays Gallery", tourism="gallery", opening_hours="Mo-Fr 10:00-17:00"),
        el("node", 16, "Grill House", amenity="restaurant"),
    ]


class FakeOSM:
    """Stands in for both services and records what was asked."""

    def __init__(self, *, nominatim=None, overpass=None, status=200, boom: Exception | None = None):
        self.nominatim = [CENTRE] if nominatim is None else nominatim
        self.overpass = {"elements": good_elements()} if overpass is None else overpass
        self.status, self.boom = status, boom
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.boom:
            raise self.boom
        if self.status != 200:
            return httpx.Response(self.status, text="nope")
        if request.url.host == "nominatim.openstreetmap.org":
            return httpx.Response(200, json=self.nominatim)
        if request.url.host == "overpass-api.de":
            return httpx.Response(200, json=self.overpass)
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    osm.clear_cache()
    monkeypatch.setattr(osm, "_throttle", lambda: None)
    yield
    osm.clear_cache()


# -------------------------------------------------------------- opening hours


@pytest.mark.parametrize("spec,expected", [
    (None, None), ("", None), ("24/7", "00:00-23:59"), ("Mo-Su 09:00-18:00", "09:00-18:00"),
    ("Mo-Fr 09:00-17:00; Sa 10:00-14:00", "10:00-14:00"), ("Sa 9:00-13:00,16:00-20:00", "09:00-13:00,16:00-20:00"),
    ("Sa,Su 10:00-18:00", "10:00-18:00"), ("Fr-Mo 10:00-18:00", "10:00-18:00"), ("Mo-Su 09:00-17:00; Sa 10:00-12:00", "10:00-12:00"),
    ("Mo-Su 09:00-18:00; PH off", "09:00-18:00"),
    ("Mo-Su 18:00-02:00", "18:00-23:59"), ("Mo-Su 00:00-24:00", "00:00-23:59"),  # past midnight: open to the end of our day
    ("Mo-Sa 09:00-18:00; Sa off", False), ("Mo-Su off", False), ("Mo-Fr 09:00-17:00", False), ("Su 10:00-18:00", False),
    ("sunrise-sunset", None), ("Mo-Su sunrise-sunset", None), ("Sa sunrise-sunset", None), ("Mo-Fr 09:00-17:00; Sa sunrise-sunset", None),
    ("by appointment", None), ("Jan-Mar Mo-Fr 09:00-17:00", None), ("Mo-Su 25:00-26:99", None), ("Mo-Su 10:00-10:10", None),
])
def test_saturday_hours(spec, expected):
    assert osm.parse_saturday_hours(spec) == expected


def test_every_hours_string_we_emit_is_one_the_scheduler_can_read():
    for spec in ["Mo-Su 18:00-02:00", "24/7", "Sa 9:00-13:00,16:00-20:00", "Mo-Su 00:00-24:00"]:
        out = osm.parse_saturday_hours(spec)
        assert isinstance(out, str)
        assert all(0 <= a < b <= 1439 for a, b in parse_windows(out))


# ------------------------------------------------------------- one element


def place(element, area="city"):
    return osm._to_place(element, area)


def test_a_park_way_uses_its_centre_and_gets_category_estimates():
    p = place(el("way", 10, "Joggers Park", lat=19.1, lon=72.9, leisure="park"))
    assert (p.id, p.kind, p.category, p.cost, p.indoor, p.source) == ("osm-w10", "activity", "park", 0, False, "openstreetmap")
    assert (p.lat, p.lon) == (19.1, 72.9) and p.diet == "na" and {"nature", "walks"} <= set(p.tags)
    assert "estimates" in p.verify


def test_stated_hours_replace_the_category_guess_and_are_labelled():
    p = place(el(tourism="museum", opening_hours="Mo-Su 10:30-16:00"))
    assert p.hours == "10:30-16:00" and p.indoor and "hours from OpenStreetMap" in p.verify
    guessed = place(el(tourism="museum"))
    assert guessed.hours == "10:00-17:30" and "hours from OpenStreetMap" not in guessed.verify


def test_venues_closed_on_saturdays_are_left_out():
    assert place(el(tourism="gallery", opening_hours="Mo-Fr 10:00-17:00")) is None
    assert place(el(tourism="gallery", opening_hours="Mo-Su 10:00-17:00; Sa off")) is None


@pytest.mark.parametrize("tags,diet,vegan", [
    ({}, "unknown", False), ({"diet:vegetarian": "only"}, "pure_veg", False), ({"diet:vegetarian": "yes"}, "veg_friendly", False),
    ({"diet:vegetarian": "no"}, "unknown", False), ({"diet:vegan": "yes"}, "unknown", True), ({"diet:vegetarian": "only", "diet:vegan": "only"}, "pure_veg", True),
])
def test_diet_tags_are_only_trusted_when_stated(tags, diet, vegan):
    p = place(el(amenity="restaurant", **tags))
    assert (p.diet, p.vegan_friendly) == (diet, vegan) and p.kind == "food" and p.indoor


def test_unknown_diet_is_a_caution_for_vegetarians_and_a_no_for_stricter_diets():
    p = place(el(amenity="cafe"))
    assert diet_check(p, "vegetarian") == (True, "Confirm they have vegetarian options")
    assert diet_check(p, "vegan") == (False, None) and diet_check(p, "jain") == (False, None)  # never guessed safe
    assert diet_check(p, "none") == (True, None)


def test_cuisine_goes_into_the_note():
    assert place(el(amenity="restaurant", cuisine="south_indian;vegetarian")).note == "Cuisine: south_indian, vegetarian."
    assert place(el(amenity="restaurant")).note == "Listed on OpenStreetMap."


@pytest.mark.parametrize("element", [
    el(name="", leisure="park"), el(name="   ", leisure="park"), el(shop="bakery"), el(leisure="pitch"), el(amenity="bank"),
    {"type": "node", "id": 5, "tags": {"name": "No coordinates", "leisure": "park"}}, {"type": "node", "id": 6},
])
def test_unusable_elements_are_skipped(element):
    assert place(element) is None


def test_names_are_cleaned_and_capped():
    p = place(el(name="<b>Lake\x00 View</b>" + "x" * 200, leisure="park"))
    assert "<" not in p.name and "\x00" not in p.name and len(p.name) <= 80


def test_historic_sites_need_no_other_tag():
    p = place(el("relation", 14, "Old Fort", historic="fort"))
    assert p.id == "osm-r14" and p.category == "historic site" and "history" in p.tags


# ---------------------------------------------------------------- fetching


def test_happy_path_builds_a_city_from_nearby_venues():
    fake = FakeOSM()
    city = osm.fetch_city("Mumbai", contact="me@example.com", client=fake.client())
    assert city and city.key == "mumbai" and city.display == "Mumbai" and city.source == "openstreetmap"
    names = {p.name for p in city.places}
    assert names == {"Joggers Park", "City Museum", "Green Leaf Cafe", "Spice Kitchen", "Old Fort", "Grill House"}  # the gallery is closed on Saturdays
    assert {p.kind for p in city.places} == {"activity", "food"}
    area = city.areas["city"]
    assert (area.lat, area.lon) == (19.076, 72.8777) and all(p.area == "city" for p in city.places)
    assert city.places[0].name == "City Museum"  # the best-documented venue comes first


def test_requests_identify_the_app_and_use_the_documented_parameters():
    fake = FakeOSM()
    osm.fetch_city("Mumbai", contact="me@example.com", client=fake.client())
    nominatim, overpass = fake.requests
    assert nominatim.method == "GET" and dict(nominatim.url.params) == {"q": "Mumbai", "format": "jsonv2", "limit": "1"}
    assert "me@example.com" in nominatim.headers["user-agent"] and "perfect-saturday-planner" in overpass.headers["user-agent"]
    assert overpass.method == "POST" and f"around:{osm.RADIUS_M},19.076,72.8777" in overpass.content.decode().replace("%3A", ":").replace("%2C", ",")


def test_the_second_lookup_comes_from_the_cache_and_clear_cache_resets_it():
    fake = FakeOSM()
    first = osm.fetch_city("Mumbai", client=fake.client())
    again = osm.fetch_city("  mumbai ", client=fake.client())
    assert again is first and len(fake.requests) == 2
    osm.clear_cache()
    osm.fetch_city("Mumbai", client=fake.client())
    assert len(fake.requests) == 4


def test_cache_entries_expire(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(osm.time, "monotonic", lambda: clock[0])
    fake = FakeOSM()
    osm.fetch_city("Mumbai", client=fake.client())
    clock[0] += osm.CACHE_TTL_S - 5
    osm.fetch_city("Mumbai", client=fake.client())
    assert len(fake.requests) == 2
    clock[0] += 10
    osm.fetch_city("Mumbai", client=fake.client())
    assert len(fake.requests) == 4


def test_a_city_osm_does_not_know_is_remembered_briefly_so_retries_do_not_hammer_it(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(osm.time, "monotonic", lambda: clock[0])
    fake = FakeOSM(nominatim=[])
    assert osm.fetch_city("Atlantis", client=fake.client()) is None
    assert osm.fetch_city("Atlantis", client=fake.client()) is None
    assert len(fake.requests) == 1
    clock[0] += osm.NEG_TTL_S + 1
    osm.fetch_city("Atlantis", client=fake.client())
    assert len(fake.requests) == 2


@pytest.mark.parametrize("fake", [
    FakeOSM(status=403), FakeOSM(status=429), FakeOSM(status=500), FakeOSM(boom=httpx.ConnectTimeout("slow")),
    FakeOSM(boom=httpx.ConnectError("no route")), FakeOSM(nominatim=[]), FakeOSM(nominatim=[{"lat": "x"}]),
    FakeOSM(nominatim=[{"nolat": 1}]), FakeOSM(overpass={"elements": []}), FakeOSM(overpass={"elements": good_elements()[:2]}),
    FakeOSM(overpass={"error": "rate limited"}), FakeOSM(overpass={"elements": [{"junk": True}, "text", 5]}),
], ids=["403", "429", "500", "timeout", "connect error", "not found", "bad lat", "missing lat", "no venues", "too few venues", "error body", "junk elements"])
def test_every_failure_is_none_never_an_exception(fake):
    assert osm.fetch_city("Mumbai", client=fake.client()) is None


def test_non_json_replies_are_survived():
    def handler(request):
        return httpx.Response(200, text="<html>maintenance</html>")

    assert osm.fetch_city("Mumbai", client=httpx.Client(transport=httpx.MockTransport(handler))) is None


def test_a_blank_name_makes_no_requests():
    fake = FakeOSM()
    assert osm.fetch_city("   ", client=fake.client()) is None and fake.requests == []


def test_results_are_capped_deduped_and_sorted_by_documentation():
    parks = [el("node", i, f"Park {i}", leisure="park") for i in range(60)]
    cafes = [el("node", 100 + i, f"Cafe {i}", amenity="cafe") for i in range(40)]
    dupes = [el("node", 200, "park 1", leisure="park"), el("node", 201, "PARK 1", leisure="garden")]
    rich = el("node", 300, "Best Documented Museum", tourism="museum", opening_hours="Mo-Su 10:00-18:00", website="w", phone="p", wheelchair="yes")
    city = osm.fetch_city("Pune Test", client=FakeOSM(overpass={"elements": [*parks, *cafes, *dupes, rich]}).client())
    acts = [p for p in city.places if p.kind == "activity"]
    foods = [p for p in city.places if p.kind == "food"]
    assert len(acts) == osm.MAX_ACTIVITIES and len(foods) == osm.MAX_FOOD
    assert len({p.name.lower() for p in city.places}) == len(city.places)
    assert city.places[0].name == "Best Documented Museum"


def test_failures_are_logged_as_warnings_with_the_reason(caplog):
    with caplog.at_level(logging.WARNING, logger="saturday.osm"):
        osm.fetch_city("Mumbai", client=FakeOSM(status=403).client())
    assert any("OSM lookup failed" in r.message and "403" in r.message for r in caplog.records)


def test_polite_throttle_spaces_nominatim_calls(monkeypatch):
    osm.clear_cache()
    monkeypatch.undo()  # use the real _throttle for this test only
    clock, slept = [100.0], []
    monkeypatch.setattr(osm.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(osm.time, "sleep", lambda s: (slept.append(round(s, 2)), clock.__setitem__(0, clock[0] + s)))
    monkeypatch.setattr(osm, "_last_nominatim", 0.0)
    osm._throttle()
    clock[0] += 0.3
    osm._throttle()
    assert slept == [0.8]


# ------------------------------------------------------- inside the agent


@pytest.fixture
def live_settings(settings):
    return dataclasses.replace(settings, enable_live_data=True)


def test_agent_plans_an_unseeded_city_from_live_data_and_labels_it(live_settings, monkeypatch):
    fake = FakeOSM()
    real = osm.fetch_city
    monkeypatch.setattr("app.agent.osm.fetch_city", lambda name, contact="": real(name, contact=contact, client=fake.client()))
    tracer = Tracer()
    res = Agent(live_settings, None).run(
        UserInput(city="Mumbai", budget=1500, available_time="3 hours", start_time="11:00 am", constraints=["vegetarian"], interests=["food", "history"]),
        tracer, allow_llm=False,
    )
    assert res.status == "ok" and res.data_source == "openstreetmap" and res.city == "Mumbai"
    step = next(e for e in tracer.events if e["tool"] == "lookupLiveVenues")
    assert step["state"] == "ok" and "OpenStreetMap" in step["summary"]
    assert res.stops and all(s.id.startswith("osm-") and s.source == "openstreetmap" for s in res.stops)
    assert all("estimates" in s.verify for s in res.stops)  # nothing from OSM is presented as verified fact
    assert res.totals.total_cost <= 1500
    cached = osm._cache["mumbai"][1]
    assert all(diet_check(cached.by_id()[s.id], "vegetarian")[0] for s in res.stops)  # the unknown-diet grill is not a vegetarian pick unless flagged
    assert not any(s.name == "Grill House" and not s.flags for s in res.stops)


def test_agent_falls_back_to_the_generic_outline_when_live_data_fails(live_settings, monkeypatch):
    fake = FakeOSM(status=403)
    real = osm.fetch_city
    monkeypatch.setattr("app.agent.osm.fetch_city", lambda name, contact="": real(name, contact=contact, client=fake.client()))
    tracer = Tracer()
    res = Agent(live_settings, None).run(UserInput(city="Mumbai", budget=1500), tracer, allow_llm=False)
    assert res.status == "fallback" and res.data_source == "generic" and "Mumbai" in res.message
    assert "OpenStreetMap lookup returned nothing usable" in res.message and "try again" in res.message  # blames the busy servers, not missing data
    step = next(e for e in tracer.events if e["tool"] == "lookupLiveVenues")
    assert step["state"] == "fallback" and "no usable live data" in step["summary"]
    assert any(e["tool"] == "fallbackPlan" for e in tracer.events)


def test_live_lookup_is_never_attempted_for_curated_cities_or_when_disabled(live_settings, settings, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("OSM must not be called")

    monkeypatch.setattr("app.agent.osm.fetch_city", forbidden)
    assert Agent(live_settings, None).run(UserInput(city="Pune", budget=1500), Tracer(), allow_llm=False).status == "ok"
    res = Agent(settings, None).run(UserInput(city="Mumbai", budget=1500), Tracer(), allow_llm=False)  # live data off
    assert res.status == "fallback" and "curated cities" in res.message and "OpenStreetMap" not in res.message


def test_the_trace_is_json_clean_when_osm_is_involved(live_settings, monkeypatch):
    fake = FakeOSM()
    real = osm.fetch_city
    monkeypatch.setattr("app.agent.osm.fetch_city", lambda name, contact="": real(name, contact=contact, client=fake.client()))
    tracer = Tracer()
    res = Agent(live_settings, None).run(UserInput(city="Mumbai", budget=2000, start_time="11:00 am"), tracer, allow_llm=False)
    json.dumps({"result": res.model_dump(mode="json"), "trace": tracer.events})


def test_a_failing_overpass_server_falls_over_to_the_next_one():
    hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == "nominatim.openstreetmap.org":
            return httpx.Response(200, json=[CENTRE])
        if request.url.host == "overpass-api.de":
            return httpx.Response(502, text="Bad Gateway")
        return httpx.Response(200, json={"elements": good_elements()})

    city = osm.fetch_city("Mumbai", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert city is not None and city.source == "openstreetmap"
    assert hosts == ["nominatim.openstreetmap.org", "overpass-api.de", "overpass.private.coffee"]


def test_when_every_source_fails_the_city_is_simply_unavailable():
    hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(200, json=[CENTRE]) if request.url.host == "nominatim.openstreetmap.org" else httpx.Response(503, text="busy")

    assert osm.fetch_city("Mumbai", client=httpx.Client(transport=httpx.MockTransport(handler))) is None
    overpass = [h for h in hosts if h != "nominatim.openstreetmap.org"]
    assert overpass == ["overpass-api.de", "overpass.private.coffee", "overpass.kumi.systems"]
    assert hosts.count("nominatim.openstreetmap.org") == 1 + len(osm.POI_PHRASES)  # the geocode, then every category search


# ----------------------------------------------------- Nominatim category search as the fallback


def poi(name, category, type_, id_=1, osm_type="node", **extra):
    """A Nominatim jsonv2 search hit, as returned with extratags=1."""
    row = {"osm_type": osm_type, "osm_id": id_, "lat": "19.07", "lon": "72.87", "category": category, "type": type_, "name": name}
    if extra:
        row["extratags"] = extra
    return row


POI_ROWS = {
    "park": [poi("Lakeside Park", "leisure", "park", 101, "way")],
    "museum": [poi("Folk Museum", "tourism", "museum", 102, opening_hours="Mo-Su 10:00-17:00")],
    "attraction": [poi("Old Palace", "tourism", "attraction", 103)],
    "cafe": [poi("Filter Coffee House", "amenity", "cafe", 104, **{"diet:vegetarian": "only"}), poi("Not A Venue", "building", "yes", 105)],
    "restaurant": [poi("Thali Kitchen", "amenity", "restaurant", 106, cuisine="indian")],
}


def test_when_overpass_is_down_venues_come_from_the_nominatim_category_search():
    phrases, overpass_hosts = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "nominatim.openstreetmap.org":
            overpass_hosts.append(request.url.host)
            return httpx.Response(502, text="Bad Gateway")
        q = request.url.params["q"]
        if q == "Mumbai":
            return httpx.Response(200, json=[CENTRE])  # the geocoding call
        phrases.append(q)
        assert request.url.params["bounded"] == "1" and request.url.params["extratags"] == "1" and "viewbox" in request.url.params
        return httpx.Response(200, json=POI_ROWS.get(q, []))

    city = osm.fetch_city("Mumbai", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert city is not None and city.source == "openstreetmap" and len(overpass_hosts) == 3
    assert phrases == list(osm.POI_PHRASES)
    by = {p.name: p for p in city.places}
    assert set(by) == {"Lakeside Park", "Folk Museum", "Old Palace", "Filter Coffee House", "Thali Kitchen"}  # the building row is not a venue
    assert by["Lakeside Park"].id == "osm-w101" and by["Lakeside Park"].kind == "activity"
    assert by["Folk Museum"].hours == "10:00-17:00" and by["Filter Coffee House"].diet == "pure_veg"
    assert all(p.source == "openstreetmap" for p in city.places)


def test_a_quick_overpass_answer_means_no_category_searches_are_made():
    phrases = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "nominatim.openstreetmap.org":
            if request.url.params["q"] != "Mumbai":
                phrases.append(request.url.params["q"])
            return httpx.Response(200, json=[CENTRE])
        return httpx.Response(200, json={"elements": good_elements()})

    city = osm.fetch_city("Mumbai", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert city is not None and phrases == []


def test_a_slow_overpass_does_not_hold_up_the_nominatim_result(monkeypatch):
    monkeypatch.setattr(osm, "HEDGE_AFTER_S", 0.05)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "nominatim.openstreetmap.org":
            time.sleep(1.5)  # a server that is up but crawling
            return httpx.Response(200, json={"elements": good_elements()})
        q = request.url.params["q"]
        return httpx.Response(200, json=[CENTRE] if q == "Mumbai" else POI_ROWS.get(q, []))

    t0 = time.monotonic()
    city = osm.fetch_city("Mumbai", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert time.monotonic() - t0 < 1.2
    assert city is not None and "Folk Museum" in {p.name for p in city.places}  # from Nominatim, not the slow Overpass answer


def test_one_failed_category_search_does_not_spoil_the_rest():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "nominatim.openstreetmap.org":
            return httpx.Response(504, text="timeout")
        q = request.url.params["q"]
        if q == "park":
            return httpx.Response(500, text="oops")
        return httpx.Response(200, json=[CENTRE] if q == "Mumbai" else POI_ROWS.get(q, []))

    city = osm.fetch_city("Mumbai", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert city is not None and "Lakeside Park" not in {p.name for p in city.places} and len(city.places) == 4


def test_markets_and_malls_are_busy_shopping_stops_that_avoid_crowds_can_filter_out():
    market = place(el("node", 30, "Old Bazaar", amenity="marketplace"))
    mall = place(el("way", 31, "City Centre Mall", shop="mall"))
    assert (market.kind, market.category, market.crowd, market.noise, market.cost) == ("activity", "market", "high", "loud", 0)
    assert {"markets", "shopping"} <= set(market.tags) and not market.indoor
    assert (mall.category, mall.crowd, mall.indoor) == ("mall", "high", True) and mall.tags == ["shopping"]
    assert place(el(shop="supermarket")) is None and place(el(shop="clothes")) is None  # only real malls count
    assert place(el(amenity="cafe")).crowd == "medium"  # everything else stays at the default

    from app.schemas import Area, CityData, Preferences
    from app.tools.options import get_activity_options
    city = CityData(key="x", display="X", avg_speed_kmph=18.0, areas={"city": Area(name="X (centre)", lat=19.07, lon=72.87)},
                    places=[market, mall, place(el("way", 32, "Joggers Park", leisure="park"))], source="openstreetmap")
    calm = get_activity_options(Preferences(avoid_crowds=True, interests=["markets", "shopping"], start_min=600, end_min=840), city)
    assert [c.place.name for c in calm.candidates] == ["Joggers Park"]
    assert {d["name"] for d in calm.skipped_notable} == {"Old Bazaar", "City Centre Mall"}
    assert all(d["reason"] == "crowded on Saturdays" for d in calm.skipped_notable)
    anything = get_activity_options(Preferences(interests=["markets", "shopping"], start_min=600, end_min=840), city)
    assert {c.place.name for c in anything.candidates} == {"Old Bazaar", "City Centre Mall", "Joggers Park"}


def test_the_overpass_query_asks_for_markets_and_malls_too():
    fake = FakeOSM()
    assert osm.fetch_city("Mumbai", client=fake.client()) is not None
    from urllib.parse import parse_qs
    body = next(r for r in fake.requests if r.url.host == "overpass-api.de").content.decode()
    query = parse_qs(body)["data"][0]
    assert "marketplace" in query and '["shop"="mall"]' in query


def test_the_category_search_also_looks_for_markets_and_malls():
    assert {"market", "shopping mall"} <= set(osm.POI_PHRASES)
    rows = [poi("Old Bazaar", "amenity", "marketplace", 41), poi("City Centre Mall", "shop", "mall", 42), poi("Corner Shop", "shop", "supermarket", 43)]
    places = [osm._to_place(e, "city") for e in map(osm._poi_to_element, rows)]
    assert [p.name if p else None for p in places] == ["Old Bazaar", "City Centre Mall", None]


def test_nominatim_hits_are_converted_to_overpass_shaped_elements_and_junk_is_dropped():
    good = osm._poi_to_element(poi("Cafe One", "amenity", "cafe", 7, "way", opening_hours="Mo-Su 08:00-22:00"))
    assert good == {"type": "way", "id": 7, "lat": 19.07, "lon": 72.87,
                    "tags": {"opening_hours": "Mo-Su 08:00-22:00", "name": "Cafe One", "amenity": "cafe"}}
    assert osm._poi_to_element({**poi("X", "amenity", "cafe"), "extratags": None})["tags"] == {"name": "X", "amenity": "cafe"}
    for junk in (None, "text", [], {}, {**poi("X", "amenity", "cafe"), "name": ""}, {**poi("X", "amenity", "cafe"), "name": None},
                 {**poi("X", "amenity", "cafe"), "lat": "north"}, {**poi("X", "amenity", "cafe"), "category": None}):
        assert osm._poi_to_element(junk) is None


def test_each_overpass_attempt_has_a_short_timeout_and_all_attempts_share_a_budget(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "nominatim.openstreetmap.org":
            return httpx.Response(200, json=[CENTRE])
        seen.append(request.extensions["timeout"]["read"])
        raise httpx.ReadTimeout("slow", request=request)

    clock = {"t": 1000.0}
    monkeypatch.setattr(osm.time, "monotonic", lambda: clock["t"])
    original = handler

    def ticking(request):  # every slow attempt uses up its whole timeout
        try:
            return original(request)
        finally:
            if request.url.host != "nominatim.openstreetmap.org":
                clock["t"] += seen[-1]

    assert osm.fetch_city("Mumbai", client=httpx.Client(transport=httpx.MockTransport(ticking))) is None
    assert all(x <= osm.PER_SERVER_S for x in seen) and sum(seen) <= osm.LOOKUP_BUDGET_S
    assert len(seen) == 3 and seen[-1] < osm.PER_SERVER_S  # the last attempt only gets what is left of the budget