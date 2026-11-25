"""HTTP layer: pages, planning endpoints, the NDJSON trace stream, and the abuse guards."""
from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app import main
from app.agent import Agent
from app.examples import EXAMPLES
from app.schemas import PlanResult
from app.util import fmt_inr

from .conftest import ASSIGNMENT, ScriptedLLM, draft, final_for
from .invariants import check_result, check_trace


@pytest.fixture
def client(monkeypatch, settings):
    """A client with fresh guards and no model, so no test depends on order or on a network."""
    monkeypatch.setattr(main, "limiter", main.RateLimiter(1000))
    monkeypatch.setattr(main, "daily", main.DailyCap(1000))
    monkeypatch.setattr(main, "_slots", threading.BoundedSemaphore(main.MAX_CONCURRENT))
    monkeypatch.setattr(main, "llm", None)
    monkeypatch.setattr(main, "agent", Agent(settings, None))
    return TestClient(main.app, raise_server_exceptions=False)


def stream(client: TestClient, body: dict, **kw) -> list[dict]:
    r = client.post("/api/plan/stream", json=body, **kw)
    assert r.status_code == 200, r.text
    return [json.loads(line) for line in r.text.splitlines() if line.strip()]


# ------------------------------------------------------------------ pages


def test_home_page_and_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Plan a Saturday that fits how you feel" in r.text and '<script src="/static/app.js"' in r.text
    assert r.headers["cache-control"] == "no-cache"
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "script-src 'self'" in csp and "frame-ancestors 'none'" in csp and "unsafe-inline" not in csp
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert "access-control-allow-origin" not in r.headers  # same-origin only


def test_there_is_no_docs_page_that_the_csp_would_turn_blank(client):
    # FastAPI's Swagger UI pulls scripts from a CDN and runs inline code; this app's CSP blocks both,
    # so the page would render empty. It is switched off; the machine-readable schema stays.
    assert client.get("/api/docs").status_code == 404 and client.get("/docs").status_code == 404
    spec = client.get("/api/openapi.json")
    assert spec.status_code == 200 and {"/api/plan", "/api/plan/stream"} <= set(spec.json()["paths"])
    html = client.get("/").text
    assert "https://" not in html and "http://" not in html and "<script>" not in html and "style=" not in html


@pytest.mark.parametrize("path,kind", [
    ("/static/app.js", "javascript"), ("/static/style.css", "css"), ("/static/favicon.svg", "svg"),
    ("/static/fonts/archivo-latin.woff2", "font"), ("/static/fonts/archivo-latin-ext.woff2", "font"),
])
def test_static_assets_are_served(client, path, kind):
    r = client.get(path)
    assert r.status_code == 200 and r.content and kind in r.headers["content-type"]


def test_unknown_paths_are_404_and_the_static_mount_is_not_a_path_traversal(client):
    assert client.get("/nope").status_code == 404
    assert client.get("/static/../main.py").status_code in {404, 400}
    assert client.get("/static/%2e%2e/main.py").status_code in {404, 400}


def test_health_endpoints(client):
    assert client.get("/healthz").json() == {"ok": True}
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["ai_enabled"] is False and h["model"] is None
    assert {c["key"] for c in h["cities"]} == {"bangalore", "pune", "gurugram"}


def test_examples_are_valid_and_every_one_plans(client):
    listed = client.get("/api/examples").json()
    assert [e["id"] for e in listed] == [e["id"] for e in EXAMPLES] and len(listed) >= 5
    for ex in listed:
        assert ex["title"] and ex["blurb"] and "<" not in ex["blurb"]
        r = client.post("/api/plan", json=ex["input"])
        assert r.status_code == 200, (ex["id"], r.text)
        body = r.json()
        assert check_result(PlanResult.model_validate(body["result"])) == [], ex["id"]


# ----------------------------------------------------------------- /api/plan


def test_plan_for_the_assignment_example(client):
    r = client.post("/api/plan", json=ASSIGNMENT)
    assert r.status_code == 200
    body = r.json()
    result = PlanResult.model_validate(body["result"])
    assert result.status == "ok" and result.mode == "basic" and result.city == "Bangalore"
    assert check_result(result) == [] and check_trace(body["trace"]) == []
    tools = {e["tool"] for e in body["trace"]}
    assert len(tools) >= 5 and {"parseUserPreferences", "getActivityOptions", "estimateCost", "validatePlan", "generateFinalPlan"} <= tools
    assert result.totals.total_cost <= 2000 and result.totals.remaining == 2000 - result.totals.total_cost
    assert fmt_inr(result.totals.total_cost) in result.summary  # the prose uses the same numbers as the totals


def test_plan_accepts_free_text_only(client):
    r = client.post("/api/plan", json={"free_text": "Pune, vegan, 3 hours, under Rs 1,500, starting at 4pm, love books"})
    assert r.status_code == 200 and r.json()["result"]["status"] == "ok"
    assert r.json()["result"]["preferences"]["diet"] == "vegan"


def test_vague_request_asks_at_most_two_questions_instead_of_guessing(client):
    body = client.post("/api/plan", json={"mood": "bored"}).json()
    assert body["result"]["status"] == "needs_info" and 1 <= len(body["result"]["questions"]) <= 2
    assert body["result"]["stops"] == []


def test_unsupported_city_degrades_to_a_labelled_generic_outline(client):
    body = client.post("/api/plan", json={"city": "Mumbai", "budget": 1500}).json()
    res = body["result"]
    assert res["status"] == "fallback" and res["data_source"] == "generic" and "Mumbai" in res["message"]
    assert all(s["id"].startswith("generic-") for s in res["stops"])
    assert any(e["tool"] == "fallbackPlan" for e in body["trace"])


@pytest.mark.parametrize("payload,field", [
    ({"city": "Pune", "budget": -1}, "budget"), ({"city": "Pune", "budget": "plenty"}, "budget"),
    ({"city": "x" * 81}, "city"), ({"free_text": "x" * 1201}, "free_text"), ({"interests": {"a": 1}}, "interests"),
])
def test_bad_fields_get_a_422_that_names_the_field(client, payload, field):
    r = client.post("/api/plan", json=payload)
    assert r.status_code == 422
    assert r.json()["error"].startswith(f"Please check '{field}'")


@pytest.mark.parametrize("body", ["not json", "[1, 2]", '"text"', "null", ""])
def test_malformed_bodies_are_422_not_500(client, body):
    r = client.post("/api/plan", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422 and "error" in r.json()


def test_unknown_fields_are_ignored(client):
    r = client.post("/api/plan", json={**ASSIGNMENT, "is_admin": True, "openai_api_key": "sk-nope"})
    assert r.status_code == 200 and "sk-nope" not in r.text


# ------------------------------------------------------------- body size cap


def test_oversized_body_is_refused_with_413(client):
    r = client.post("/api/plan", content=b'{"free_text": "' + b"x" * 20_000 + b'"}', headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json() == {"error": "That request is too large."}
    assert r.headers["x-content-type-options"] == "nosniff"  # even the refusal carries the security headers


def test_chunked_upload_without_content_length_is_also_capped(client):
    def chunks():
        yield b'{"free_text": "'
        for _ in range(40):
            yield b"x" * 1024
        yield b'"}'

    r = client.post("/api/plan", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json()["error"] == "That request is too large."


def test_the_cap_is_exact(client):
    raw = json.dumps(ASSIGNMENT)
    headers = {"content-type": "application/json"}
    at_limit = client.post("/api/plan", content=raw + " " * (main.MAX_BODY - len(raw)), headers=headers)
    over = client.post("/api/plan", content=raw + " " * (main.MAX_BODY - len(raw) + 1), headers=headers)
    assert at_limit.status_code == 200 and over.status_code == 413


def test_bad_content_length_is_a_400(client):
    r = client.post("/api/plan", content=b"{}", headers={"content-type": "application/json", "content-length": "abc"})
    assert r.status_code in {400, 422}


# ------------------------------------------------------------- rate limiting


def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(main, "limiter", main.RateLimiter(2))
    assert [client.post("/api/plan", json=ASSIGNMENT).status_code for _ in range(2)] == [200, 200]
    r = client.post("/api/plan", json=ASSIGNMENT)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert "wait" in r.json()["error"]
    assert client.get("/healthz").status_code == 200  # only planning is limited


def test_rate_limit_keys_on_the_last_forwarded_hop_so_it_cannot_be_spoofed(client, monkeypatch):
    monkeypatch.setattr(main, "limiter", main.RateLimiter(1))
    spoof = lambda first: {"x-forwarded-for": f"{first}, 10.0.0.9"}  # noqa: E731 - the proxy appends the real peer last
    assert client.post("/api/plan", json=ASSIGNMENT, headers=spoof("1.1.1.1")).status_code == 200
    assert client.post("/api/plan", json=ASSIGNMENT, headers=spoof("2.2.2.2")).status_code == 429  # same real peer
    other = {"x-forwarded-for": "1.1.1.1, 10.0.0.10"}
    assert client.post("/api/plan", json=ASSIGNMENT, headers=other).status_code == 200  # a different peer is unaffected


def test_rate_limit_window_slides(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: now[0])
    rl = main.RateLimiter(2)
    assert rl.check("a") == (True, 0) and rl.check("a") == (True, 0)
    ok, retry = rl.check("a")
    assert not ok and 1 <= retry <= 61
    assert rl.check("b") == (True, 0)  # independent keys
    now[0] += 61
    assert rl.check("a") == (True, 0)


def test_rate_limiter_forgets_idle_clients_instead_of_growing_forever(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: now[0])
    rl = main.RateLimiter(5)
    for i in range(5001):
        rl.check(f"k{i}")
    now[0] += 120
    rl.check("fresh")
    assert len(rl._hits) < 100


def test_busy_server_says_so_with_503_and_recovers(client, monkeypatch):
    sem = threading.BoundedSemaphore(1)
    sem.acquire()
    monkeypatch.setattr(main, "_slots", sem)
    for path in ("/api/plan", "/api/plan/stream"):
        r = client.post(path, json=ASSIGNMENT)
        assert r.status_code == 503 and "busy" in r.json()["error"]
    sem.release()
    assert client.post("/api/plan", json=ASSIGNMENT).status_code == 200


# -------------------------------------------------------------- failures


def test_internal_errors_are_a_clean_500_that_hides_the_cause_and_frees_the_slot(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(main.agent, "run", boom)
    for _ in range(main.MAX_CONCURRENT + 2):  # would starve the pool if a failed request kept its slot
        r = client.post("/api/plan", json=ASSIGNMENT)
        assert r.status_code == 500 and r.json() == {"error": "Something went wrong on my side. Please try again."}
        assert "secret" not in r.text


def test_stream_reports_internal_errors_as_an_error_event(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(main.agent, "run", boom)
    events = stream(client, ASSIGNMENT)
    assert events[-1] == {"type": "error", "message": "Something went wrong on my side. Please try again."}
    assert "secret" not in json.dumps(events)


# ------------------------------------------------------------------ stream


def test_stream_emits_each_step_as_running_then_done_and_ends_with_the_result(client):
    events = stream(client, ASSIGNMENT)
    steps = [e for e in events if e["type"] == "step"]
    assert events[-1]["type"] == "result" and all(e["type"] in {"step", "ping", "result"} for e in events)
    by_id: dict[int, list[str]] = {}
    for e in steps:
        by_id.setdefault(e["id"], []).append(e["state"])
    assert all(states[0] == "running" and states[-1] != "running" and len(states) == 2 for states in by_id.values())
    final = events[-1]
    assert [t["id"] for t in final["trace"]] == sorted(by_id)  # the result carries the same trace the stream showed
    assert check_result(PlanResult.model_validate(final["result"])) == []


def test_stream_and_plain_endpoints_agree(client):
    plain = client.post("/api/plan", json=ASSIGNMENT).json()["result"]
    streamed = stream(client, ASSIGNMENT)[-1]["result"]
    for key in ("status", "headline", "summary", "totals", "stops", "trade_offs", "cautions", "assumptions"):
        assert plain[key] == streamed[key], key


def test_stream_validation_errors_are_plain_422_json(client):
    r = client.post("/api/plan/stream", json={"city": "Pune", "budget": -3})
    assert r.status_code == 422 and r.headers["content-type"].startswith("application/json")


def test_stream_has_the_headers_that_keep_proxies_from_buffering(client):
    r = client.post("/api/plan/stream", json=ASSIGNMENT)
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert r.headers["cache-control"] == "no-cache" and r.headers["x-accel-buffering"] == "no"


def test_stream_sends_a_heartbeat_while_a_slow_step_runs(client, monkeypatch):
    real = main.agent.run
    monkeypatch.setattr(main.asyncio, "wait_for", _fast_timeout(main.asyncio.wait_for))

    def slow(*a, **k):
        time.sleep(0.25)
        return real(*a, **k)

    monkeypatch.setattr(main.agent, "run", slow)
    events = stream(client, ASSIGNMENT)
    assert any(e["type"] == "ping" for e in events) and events[-1]["type"] == "result"


def _fast_timeout(real_wait_for):
    async def wait_for(aw, timeout):
        return await real_wait_for(aw, timeout=0.05)

    return wait_for


# ------------------------------------------------------- model on, with a daily cap


def test_daily_cap_switches_to_basic_mode_and_says_so(monkeypatch, settings):
    llm = ScriptedLLM(
        parse={"energy": "low"}, draft=draft("blr-cubbon", "blr-adigas", "blr-tipu"),
        final=final_for(["blr-cubbon", "blr-adigas", "blr-tipu"]),
    )
    monkeypatch.setattr(main, "limiter", main.RateLimiter(1000))
    monkeypatch.setattr(main, "daily", main.DailyCap(1))
    monkeypatch.setattr(main, "_slots", threading.BoundedSemaphore(4))
    monkeypatch.setattr(main, "llm", llm)
    monkeypatch.setattr(main, "agent", Agent(settings, llm))
    c = TestClient(main.app, raise_server_exceptions=False)
    first = c.post("/api/plan", json=ASSIGNMENT).json()["result"]
    second = c.post("/api/plan", json=ASSIGNMENT).json()["result"]
    assert first["mode"] == "ai" and first["llm"]["enabled"] and first["llm"]["calls"] >= 1
    assert second["mode"] == "basic" and second["status"] == "ok"
    assert second["llm"]["notes"][0].startswith("Today's AI quota")
    calls_after_cap = llm.count("draft")
    c.post("/api/plan", json=ASSIGNMENT)
    assert llm.count("draft") == calls_after_cap  # no further model calls once the cap is hit


def test_daily_cap_rolls_over_at_a_new_utc_day():
    cap = main.DailyCap(2)
    assert [cap.take() for _ in range(3)] == [True, True, False]
    cap._day = "1999-12-31"
    assert cap.take() is True and cap._n == 1
