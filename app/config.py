"""Runtime settings, read from environment variables (or a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

try:  # optional: lets `.env` work for local runs
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - python-dotenv is optional
    pass


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- LLM -----------------------------------------------------------
    openai_api_key: str | None = None
    # Cheapest current OpenAI tier at time of writing. Override with OPENAI_MODEL.
    openai_model: str = "gpt-6-luna"
    openai_base_url: str | None = None  # any OpenAI-compatible endpoint
    openai_api_mode: str = "chat"  # "chat" (Chat Completions) or "responses"
    # "auto": send a small per-task reasoning effort to reasoning models only;
    # "off": never send it; or force one value: none|low|medium|high
    reasoning_effort: str = "auto"
    llm_timeout_s: float = 25.0
    llm_max_output_tokens: int = 2500
    # --- agent ---------------------------------------------------------
    max_repairs: int = 2  # validate -> redraft loops before falling back
    request_deadline_s: float = 75.0  # after this, stop calling the LLM
    # --- abuse protection ---------------------------------------------
    rate_limit_per_min: int = 20  # per client address
    daily_llm_run_cap: int = 400  # runs that may use the LLM per UTC day
    # --- data ----------------------------------------------------------
    enable_live_data: bool = True  # OpenStreetMap lookup for unseeded cities
    contact_email: str = ""  # polite contact for the Nominatim User-Agent
    log_level: str = "INFO"

    @property
    def ai_enabled(self) -> bool:
        return bool(self.openai_api_key)


def load_settings() -> Settings:
    return Settings(
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
        openai_model=(os.getenv("OPENAI_MODEL") or "gpt-6-luna").strip(),
        openai_base_url=(os.getenv("OPENAI_BASE_URL") or "").strip() or None,
        openai_api_mode=(os.getenv("OPENAI_API_MODE") or "chat").strip().lower(),
        reasoning_effort=(os.getenv("OPENAI_REASONING_EFFORT") or "auto").strip().lower(),
        llm_timeout_s=_float("LLM_TIMEOUT_S", 25.0),
        llm_max_output_tokens=_int("LLM_MAX_OUTPUT_TOKENS", 2500),
        max_repairs=_int("MAX_REPAIRS", 2),
        request_deadline_s=_float("REQUEST_DEADLINE_S", 75.0),
        rate_limit_per_min=_int("RATE_LIMIT_PER_MIN", 20),
        daily_llm_run_cap=_int("DAILY_LLM_RUN_CAP", 400),
        enable_live_data=_bool("ENABLE_LIVE_DATA", True),
        contact_email=(os.getenv("CONTACT_EMAIL") or "").strip(),
        log_level=(os.getenv("LOG_LEVEL") or "INFO").strip().upper(),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
