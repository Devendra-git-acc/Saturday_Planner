"""HTTP layer: serves the UI and the planning API (plain JSON or a live NDJSON trace stream)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import Agent
from .cities import supported_cities
from .config import get_settings
from .examples import EXAMPLES
from .llm import build_llm
from .schemas import UserInput
from .trace import Tracer

settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("saturday.api")

STATIC = Path(__file__).parent / "static"
MAX_BODY = 16 * 1024
MAX_CONCURRENT = 8

# No interactive docs page: Swagger UI loads scripts from a CDN and runs inline code, both of which the strict
# Content-Security-Policy below blocks, so it would render as a blank page. The schema stays at /api/openapi.json.
app = FastAPI(title="Perfect Saturday Planner", docs_url=None, redoc_url=None, openapi_url="/api/openapi.json")
llm = build_llm(settings)
agent = Agent(settings, llm)


# ------------------------------------------------------------- guards


class RateLimiter:
    """Sliding-window limiter per client key. In-memory: fine for one small instance."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > 5000:
                self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < 60}
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False, max(1, int(60 - (now - q[0])) + 1)
            q.append(now)
            return True, 0


class DailyCap:
    """Caps model-assisted runs per UTC day so a public demo cannot run up the bill."""

    def __init__(self, cap: int) -> None:
        self.cap, self._day, self._n = cap, "", 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if today != self._day:
                self._day, self._n = today, 0
            if self._n >= self.cap:
                return False
            self._n += 1
            return True


limiter = RateLimiter(settings.rate_limit_per_min)
daily = DailyCap(settings.daily_llm_run_cap)
_slots = threading.BoundedSemaphore(MAX_CONCURRENT)


def client_key(request: Request) -> str:
    """Last X-Forwarded-For hop = the address our (single) trusted proxy saw; else the socket peer."""
    xff = request.headers.get("x-forwarded-for", "")
    ip = xff.split(",")[-1].strip() if xff else (request.client.host if request.client else "unknown")
    return hashlib.sha256(ip.encode()).hexdigest()[:12]


class BodyLimit:
    """Rejects POST bodies over `max_bytes`, whether or not the client declared a Content-Length.

    A declared size is checked up front; for chunked uploads the bytes are counted as the app reads
    them, the 413 is sent the moment the limit is crossed, and the app is told the client went away.
    """

    def __init__(self, app, max_bytes: int) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)

        started = refused = False
        seen = 0

        async def reply(status: int, message: str) -> None:
            body = json.dumps({"error": message}).encode()
            await send({"type": "http.response.start", "status": status, "headers": [
                (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    return await reply(413, "That request is too large.")
            except ValueError:
                return await reply(400, "Bad request.")

        async def counted():
            nonlocal seen, refused
            if refused:
                return {"type": "http.disconnect"}
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.max_bytes:
                    refused = True
                    if not started:
                        await reply(413, "That request is too large.")
                    return {"type": "http.disconnect"}
            return msg

        async def watch(msg):
            nonlocal started
            if refused:
                return  # we already answered; drop whatever the app says next
            started = started or msg["type"] == "http.response.start"
            await send(msg)

        await self.app(scope, counted, watch)


app.add_middleware(BodyLimit, max_bytes=MAX_BODY)  # added first, so it sits inside `guard` and gets its headers


@app.middleware("http")
async def guard(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    )
    return resp


@app.exception_handler(RequestValidationError)
async def bad_input(request: Request, exc: RequestValidationError):
    err = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(x) for x in err.get("loc", [])[1:]) or "request"
    msg = str(err.get("msg", "invalid value")).removeprefix("Value error, ")
    return JSONResponse({"error": f"Please check '{field}': {msg}"}, status_code=422)


# ------------------------------------------------------------- pages / info


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"ok": True}


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "ai_enabled": llm is not None,
        "model": getattr(llm, "model", None),
        "live_data": settings.enable_live_data,
        "cities": supported_cities(),
    }


@app.get("/api/examples")
def examples() -> list[dict]:
    return EXAMPLES


# ------------------------------------------------------------- planning


def _prepare(request: Request) -> tuple[bool, JSONResponse | None]:
    ok, retry = limiter.check(client_key(request))
    if not ok:
        return False, JSONResponse(
            {"error": f"You're going a bit fast - please wait {retry} seconds and try again."},
            status_code=429, headers={"Retry-After": str(retry)},
        )
    if not _slots.acquire(blocking=False):
        return False, JSONResponse({"error": "I'm busy planning other outings. Try again in a few seconds."}, status_code=503)
    return True, None


def _run(inp: UserInput, tracer: Tracer):
    allow_llm = llm is not None and daily.take()
    result = agent.run(inp, tracer, allow_llm=allow_llm)
    if llm is not None and not allow_llm:
        result.llm.notes.insert(0, "Today's AI quota for this demo is used up, so this plan used basic mode.")
    log.info(
        "plan status=%s mode=%s city=%s stops=%d ms=%d llm_calls=%d tokens=%d",
        result.status, result.mode, result.city, len(result.stops), result.elapsed_ms, result.llm.calls, result.llm.tokens,
    )
    return result


@app.post("/api/plan")
def plan(inp: UserInput, request: Request):
    ok, resp = _prepare(request)
    if not ok:
        return resp
    try:
        tracer = Tracer()
        result = _run(inp, tracer)
        return {"result": result.model_dump(mode="json"), "trace": tracer.events}
    except Exception:  # noqa: BLE001
        log.exception("plan failed")
        return JSONResponse({"error": "Something went wrong on my side. Please try again."}, status_code=500)
    finally:
        _slots.release()


@app.post("/api/plan/stream")
async def plan_stream(inp: UserInput, request: Request):
    ok, resp = _prepare(request)
    if not ok:
        return resp
    loop = asyncio.get_running_loop()
    aq: asyncio.Queue = asyncio.Queue()
    done = object()

    def emit(item) -> None:
        loop.call_soon_threadsafe(aq.put_nowait, item)

    def worker() -> None:
        try:
            tracer = Tracer(on_event=emit)
            result = _run(inp, tracer)
            emit({"type": "result", "result": result.model_dump(mode="json"), "trace": tracer.events})
        except Exception:  # noqa: BLE001
            log.exception("stream plan failed")
            emit({"type": "error", "message": "Something went wrong on my side. Please try again."})
        finally:
            _slots.release()
            emit(done)

    try:
        threading.Thread(target=worker, daemon=True).start()
    except RuntimeError:  # cannot start a thread right now: give the slot back and say so
        log.exception("could not start worker thread")
        _slots.release()
        return JSONResponse({"error": "I'm busy planning other outings. Try again in a few seconds."}, status_code=503)

    async def gen():
        while True:
            try:
                item = await asyncio.wait_for(aq.get(), timeout=8)
            except asyncio.TimeoutError:
                yield b'{"type":"ping"}\n'  # heartbeat: keeps proxies from closing a quiet stream
                continue
            if item is done:
                return
            yield (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(
        gen(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
