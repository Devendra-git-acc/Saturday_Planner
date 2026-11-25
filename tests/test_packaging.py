"""Things that only break once the app is deployed somewhere that is not this machine."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_NAME = {"dotenv": "python-dotenv"}  # import name -> pip name
COMES_WITH_FASTAPI = {"starlette"}


def _declared() -> dict[str, str]:
    out = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, _, version = line.partition("==")
            out[name.strip().lower().replace("_", "-")] = version.strip()
    return out


def _imported_third_party() -> set[str]:
    found: set[str] = set()
    for path in (ROOT / "app").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return {n for n in found if n not in sys.stdlib_module_names and n != "app"}


def test_every_package_the_app_imports_is_in_requirements():
    """A clean install on the host must be enough; a package that only exists in a dev environment is a deploy bug."""
    declared = set(_declared())
    missing = {DIST_NAME.get(n, n) for n in _imported_third_party() if n not in COMES_WITH_FASTAPI} - declared
    assert not missing, f"imported but not in requirements.txt: {sorted(missing)}"


def test_requirements_are_pinned_and_the_server_is_included():
    pins = _declared()
    assert all(re.fullmatch(r"\d+(\.\d+)+", v) for v in pins.values()), pins
    assert {"fastapi", "uvicorn", "openai", "pydantic", "httpx"} <= set(pins)


def test_env_example_documents_exactly_the_variables_the_app_reads():
    read = set(re.findall(r'(?:getenv|_int|_bool|_float)\(\s*"([A-Z_]+)"', (ROOT / "app" / "config.py").read_text()))
    documented = set(re.findall(r"^([A-Z_]+)=", (ROOT / ".env.example").read_text(), flags=re.M))
    assert read, "could not find any environment variables in config.py"
    assert read == documented, f"only in code: {sorted(read - documented)}; only in .env.example: {sorted(documented - read)}"


def test_render_blueprint_matches_the_app():
    text = (ROOT / "render.yaml").read_text()
    assert "healthCheckPath: /healthz" in text and "app.main:app" in text and "$PORT" in text
    assert "sync: false" in text and not re.search(r"OPENAI_API_KEY\s*\n\s*value", text)  # the key is entered in the dashboard only
    assert "OPENAI_API_KEY=sk-" not in text


def test_no_secret_shaped_strings_are_committed():
    pattern = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
    offenders = [
        str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
        if p.is_file() and p.suffix in {".py", ".md", ".yaml", ".yml", ".txt", ".example", ".js", ".html", ".json", ""}
        and ".pytest_cache" not in p.parts and "__pycache__" not in p.parts and p.stat().st_size < 1_000_000 and pattern.search(p.read_text(errors="ignore"))
    ]
    assert not offenders, offenders


def test_the_dockerfile_runs_the_same_command_as_the_procfile():
    docker, proc = (ROOT / "Dockerfile").read_text(), (ROOT / "Procfile").read_text()
    for text in (docker, proc):
        assert "app.main:app" in text and "--host 0.0.0.0" in text
    assert "USER planner" in docker and "COPY app ./app" in docker
