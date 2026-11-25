"""The README is part of the submission, so it must not drift from the code it describes."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import load_settings

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
ROOT_FILES = {"render.yaml", "Procfile", "Dockerfile", ".env.example", ".python-version", "requirements.txt", "requirements-dev.txt"}


def _section(title: str) -> str:
    """Text from a '## title' or '### title' heading up to the next heading of the same or higher level."""
    m = re.search(rf"^(#+) {re.escape(title)}\s*$", README, flags=re.M)
    assert m, f"README has no '{title}' heading"
    level = len(m.group(1))
    rest = README[m.end():]
    nxt = re.search(rf"^#{{1,{level}}} ", rest, flags=re.M)
    return rest[: nxt.start()] if nxt else rest


def test_files_and_scripts_the_readme_mentions_exist():
    spans = set(re.findall(r"`([^`\n]+)`", README))
    paths = {s for s in spans if re.fullmatch(r"(app|scripts|tests|docs)/[\w./-]+", s) or s in ROOT_FILES}
    paths |= set(re.findall(r"\]\(((?!https?:|#)[^)\s]+)\)", README))  # relative links and images
    paths |= set(re.findall(r"python (scripts/[\w./-]+\.py)", README))
    assert paths >= {"render.yaml", "docs/screenshot-plan.png", "docs/screenshot-trace.png", "scripts/smoke_llm.py"}
    missing = sorted(p for p in paths if not (ROOT / p).exists())
    assert not missing, f"README mentions files that do not exist: {missing}"


def test_every_setting_in_the_readme_table_is_read_by_the_code_and_has_the_same_default():
    config_src = (ROOT / "app" / "config.py").read_text()
    read = set(re.findall(r'(?:getenv|_int|_bool|_float)\(\s*"([A-Z_]+)"', config_src))
    documented: dict[str, str] = {}
    for row in re.findall(r"^\|(.+)\|\s*$", _section("Configuration"), flags=re.M):
        cells = [c.strip() for c in row.split("|")]
        if len(cells) < 3 or set(cells[0]) <= {"-", " "} or cells[0] == "Variable":
            continue
        names = re.findall(r"`([A-Z][A-Z_]+)`", cells[0])
        defaults = [d.strip().strip("`") for d in cells[1].split(",")] if cells[1] else []
        for i, name in enumerate(names):
            documented[name] = defaults[i] if i < len(defaults) else ""
    assert set(documented) == read, f"only in code: {sorted(read - set(documented))}; only in README: {sorted(set(documented) - read)}"

    # Compare defaults with what the code really uses when nothing is set.
    attr = {
        "OPENAI_MODEL": "openai_model", "OPENAI_API_MODE": "openai_api_mode", "OPENAI_REASONING_EFFORT": "reasoning_effort",
        "LLM_TIMEOUT_S": "llm_timeout_s", "LLM_MAX_OUTPUT_TOKENS": "llm_max_output_tokens", "MAX_REPAIRS": "max_repairs",
        "REQUEST_DEADLINE_S": "request_deadline_s", "RATE_LIMIT_PER_MIN": "rate_limit_per_min",
        "DAILY_LLM_RUN_CAP": "daily_llm_run_cap", "ENABLE_LIVE_DATA": "enable_live_data",
    }
    import os

    saved = {k: os.environ.pop(k) for k in list(os.environ) if k in read}
    try:
        defaults = load_settings()
    finally:
        os.environ.update(saved)
    for name, attribute in attr.items():
        actual, shown = getattr(defaults, attribute), documented[name]
        same = float(shown) == float(actual) if re.fullmatch(r"[\d.]+", shown) else str(actual).lower() == shown.lower()
        assert same, f"{name}: README says {shown!r}, code uses {actual!r}"


def test_every_tool_the_readme_lists_is_a_real_step_in_the_agent():
    agent_src = (ROOT / "app" / "agent.py").read_text()
    names = re.findall(r"^\| `(\w+)` \|", _section("The tools"), flags=re.M)
    assert len(names) >= 10, names
    assert {"parseUserPreferences", "getActivityOptions", "getFoodOptions", "draftPlan", "estimateCost", "validatePlan", "generateFinalPlan"} <= set(names)
    missing = [n for n in names if f'"{n}"' not in agent_src]
    assert not missing, f"README lists tools the agent never runs: {missing}"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "llm", None)
    return TestClient(main.app)


def test_every_endpoint_the_readme_documents_exists(client):
    spec = client.get("/api/openapi.json").json()["paths"]
    rows = re.findall(r"^\| `(GET|POST) ([^`]+)`", _section("API"), flags=re.M)
    rows += [(m, p.strip()) for m, p in re.findall(r"`(GET) (/[^`,]+)`", _section("API"))]
    assert {("POST", "/api/plan"), ("POST", "/api/plan/stream"), ("GET", "/api/health")} <= set(rows)
    for method, path in set(rows):
        if path in spec:
            assert method.lower() in spec[path], f"{method} {path} is documented but the app does not serve it"
        else:
            assert method == "GET" and client.get(path).status_code == 200, f"{method} {path} is documented but missing"


def test_the_readme_example_request_is_a_valid_plan_request(client):
    import json

    block = re.search(r"curl -s localhost:8000/api/plan[^\n]*-d '(\{.*?\})'", README, flags=re.S)
    assert block, "README has no curl example for /api/plan"
    body = json.loads(block.group(1))
    r = client.post("/api/plan", json=body)
    assert r.status_code == 200 and r.json()["result"]["status"] == "ok" and r.json()["trace"]


def test_the_readme_venue_counts_match_the_data():
    from app.cities import get_city

    n = {k: len(get_city(k).places) for k in ("bangalore", "pune", "gurugram")}
    assert f"Bangalore ({n['bangalore']} venues), Pune ({n['pune']}) and Gurugram ({n['gurugram']})" in README
