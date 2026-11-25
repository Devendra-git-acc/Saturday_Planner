"""Data shapes used across the agent. Everything the LLM returns is validated against these."""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from .util import to_number

Level = Literal["low", "medium", "high"]
Energy = Level
Social = Literal["quiet", "balanced", "lively"]
Diet = Literal["none", "vegetarian", "vegan", "jain"]

# The only interest words the rest of the system understands.
INTEREST_VOCAB = [
    "food", "coffee", "music", "walks", "nature", "art", "history", "books", "markets",
    "shopping", "nightlife", "theatre", "movies", "photography", "wellness", "spirituality", "games", "science",
]


def _as_list(v: Any) -> Any:
    if v is None:
        return []
    if isinstance(v, str):
        return [p.strip() for p in re.split(r"[,;\n]", v) if p.strip()]
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return v


# ------------------------------------------------------------------- input


class UserInput(BaseModel):
    """What the API accepts. Structured fields and/or free text - both are fine."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    city: str | None = Field(default=None, max_length=80)
    budget: float | None = Field(default=None, ge=0, le=1_000_000)
    available_time: str | None = Field(default=None, max_length=60)
    mood: str | None = Field(default=None, max_length=200)
    interests: list[str] = Field(default_factory=list, max_length=12)
    constraints: list[str] = Field(default_factory=list, max_length=12)
    start_time: str | None = Field(default=None, max_length=20)
    free_text: str | None = Field(default=None, max_length=1200)

    @field_validator("city", "available_time", "mood", "start_time", "free_text", mode="before")
    @classmethod
    def _scalar_to_str(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = str(v)
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("budget", mode="before")
    @classmethod
    def _budget(cls, v: Any) -> Any:
        return to_number(v)

    @field_validator("interests", "constraints", mode="before")
    @classmethod
    def _lists(cls, v: Any) -> Any:
        v = _as_list(v)
        if isinstance(v, list):
            return [x[:60] for x in v][:12]
        return v


# ------------------------------------------------------------------- places


class Area(BaseModel):
    name: str
    lat: float
    lon: float


class Place(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    kind: Literal["activity", "food"]
    category: str = ""
    area: str = ""
    tags: list[str] = Field(default_factory=list)
    cost: int = 0  # approx. INR per person
    duration: int = 60  # typical minutes
    min_duration: int | None = None
    max_duration: int | None = None
    hours: str = "09:00-21:00"  # Saturday windows, 'HH:MM-HH:MM[,HH:MM-HH:MM]'
    live_only: bool = False  # the draw (live music) only happens inside `hours`; outside them it is at best a plain cafe or bar
    energy: Level = "low"
    crowd: Level = "medium"  # usual Saturday level outside peak windows
    peak: list[str] = Field(default_factory=list)  # windows that bump the crowd one level
    noise: Literal["quiet", "moderate", "loud"] = "moderate"
    indoor: bool = False
    diet: Literal["pure_veg", "veg_friendly", "non_veg_focus", "unknown", "na"] = "na"
    vegan_friendly: bool = False
    jain_friendly: bool = False
    alcohol_centric: bool = False
    dairy_centric: bool = False  # milk/ghee is the point of the place (so not vegan-adaptable)
    meal: list[str] = Field(default_factory=list)
    note: str = ""
    verify: str = ""  # shown to the user as a caution
    lat: float | None = None
    lon: float | None = None
    source: str = "curated"

    @property
    def min_dur(self) -> int:
        return self.min_duration or max(15, self.duration // 2)

    @property
    def max_dur(self) -> int:
        return self.max_duration or int(self.duration * 1.6)


class CityData(BaseModel):
    key: str
    display: str
    aliases: list[str] = Field(default_factory=list)
    avg_speed_kmph: float = 18.0  # typical Saturday road speed between areas
    areas: dict[str, Area] = Field(default_factory=dict)
    places: list[Place] = Field(default_factory=list)
    source: str = "curated"  # curated | openstreetmap | generic

    _km_cache: dict = PrivateAttr(default_factory=dict)

    def by_id(self) -> dict[str, Place]:
        return {p.id: p for p in self.places}


# -------------------------------------------------------------- preferences


class Preferences(BaseModel):
    city_input: str | None = None
    city: str | None = None  # canonical key (bangalore|pune|gurugram) or normalised raw text
    city_display: str | None = None
    city_known: bool = False  # True when a curated dataset exists
    city_ambiguous: bool = False  # several supported cities were named; ask rather than pick
    budget_inr: int | None = None
    minutes: int | None = None
    start_min: int | None = None  # minutes after midnight
    end_min: int | None = None  # start + minutes, capped at 23:59
    energy: Energy = "medium"
    social: Social = "balanced"
    interests: list[str] = Field(default_factory=list)
    diet: Diet = "none"
    avoid_crowds: bool = False
    avoid_alcohol: bool = False
    indoor_only: bool = False
    other_constraints: list[str] = Field(default_factory=list)  # things we cannot verify
    mood_summary: str = ""
    assumptions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    parsed_by: Literal["llm", "rules"] = "rules"
    defaulted: list[str] = Field(default_factory=list)  # inputs we had to assume: budget|time|interests|start


# ------------------------------------------------------------------ options


class Candidate(BaseModel):
    place: Place
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)  # e.g. 'ask for vegan preparation'


class OptionsResult(BaseModel):
    candidates: list[Candidate] = Field(default_factory=list)
    dropped: dict[str, int] = Field(default_factory=dict)  # reason -> count
    skipped_notable: list[dict] = Field(default_factory=list)  # [{'name', 'reason'}]
    pool_size: int = 0


# --------------------------------------------------------------------- plan


class DraftStop(BaseModel):
    place_id: str
    duration_min: int | None = None
    reason: str = ""


class DraftPlan(BaseModel):
    stops: list[DraftStop] = Field(default_factory=list)
    notes: str = ""


class Leg(BaseModel):
    minutes: int = 0
    cost: int = 0
    mode: str = "start"
    km: float = 0.0


class ScheduledStop(BaseModel):
    id: str
    name: str
    kind: str
    category: str = ""
    area_name: str = ""
    start: int
    end: int
    duration: int
    wait: int = 0
    cost: int = 0
    travel: Leg = Field(default_factory=Leg)
    crowd_level: Level = "medium"
    closed: bool = False
    energy: Level = "low"
    tags: list[str] = Field(default_factory=list)
    note: str = ""
    verify: str = ""
    diet: str = "na"
    flags: list[str] = Field(default_factory=list)
    hours: str = ""
    source: str = "curated"
    why: str = ""
    warnings: list[str] = Field(default_factory=list)


class Issue(BaseModel):
    code: str
    severity: Literal["error", "warn", "info"]
    message: str
    stop_id: str | None = None
    major: bool = False  # worth another LLM attempt even though the plan is valid
    user_message: str | None = None  # reader-friendly wording when `message` is written as an instruction to the planner


class CostReport(BaseModel):
    stops: list[ScheduledStop] = Field(default_factory=list)
    spend: int = 0
    travel_cost: int = 0
    total_cost: int = 0
    start: int = 0
    end: int = 0
    total_minutes: int = 0
    travel_minutes: int = 0
    pre_issues: list[Issue] = Field(default_factory=list)  # unknown/duplicate stops


class ValidationReport(BaseModel):
    ok: bool = True  # no hard errors
    clean: bool = True  # no errors and nothing worth repairing
    issues: list[Issue] = Field(default_factory=list)
    penalty: float = 0.0

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warn"]

    @property
    def repair_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error" or (i.severity == "warn" and i.major)]


# ------------------------------------------------- LLM output (validated)


def _norm_enum(v: Any, allowed: tuple[str, ...], default: str, aliases: dict[str, str] | None = None) -> str:
    if v is None:
        return default
    s = str(v).strip().lower()
    if aliases and s in aliases:
        s = aliases[s]
    return s if s in allowed else default


class LLMPrefs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    city: str | None = None
    budget_inr: float | None = None
    minutes: float | None = None
    start_time: str | None = None
    energy: str = "medium"
    social: str = "balanced"
    interests: list[str] = Field(default_factory=list)
    diet: str = "none"
    avoid_crowds: bool = False
    avoid_alcohol: bool = False
    indoor_only: bool = False
    other_constraints: list[str] = Field(default_factory=list)
    mood_summary: str = ""

    @field_validator("budget_inr", "minutes", mode="before")
    @classmethod
    def _num(cls, v: Any) -> Any:
        try:
            return to_number(v)
        except ValueError:
            return None

    @field_validator("interests", "other_constraints", mode="before")
    @classmethod
    def _lists(cls, v: Any) -> Any:
        return _as_list(v)

    @field_validator("energy", mode="after")
    @classmethod
    def _energy(cls, v: str) -> str:
        return _norm_enum(v, ("low", "medium", "high"), "medium", {"tired": "low", "normal": "medium", "energetic": "high"})

    @field_validator("social", mode="after")
    @classmethod
    def _social(cls, v: str) -> str:
        return _norm_enum(v, ("quiet", "balanced", "lively"), "balanced", {"calm": "quiet", "social": "lively"})

    @field_validator("diet", mode="after")
    @classmethod
    def _diet(cls, v: str) -> str:
        return _norm_enum(v, ("none", "vegetarian", "vegan", "jain"), "none", {"veg": "vegetarian", "non-vegetarian": "none", "anything": "none"})


class LLMStop(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    duration_min: int | None = None
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, v: Any) -> Any:
        if isinstance(v, dict):
            v = dict(v)
            if "id" not in v:
                for k in ("place_id", "placeId", "venue_id"):
                    if k in v:
                        v["id"] = v[k]
                        break
            if "duration_min" not in v:
                for k in ("duration", "minutes", "duration_minutes"):
                    if k in v:
                        v["duration_min"] = v[k]
                        break
        return v

    @field_validator("duration_min", mode="before")
    @classmethod
    def _dur(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        try:
            return int(round(float(v)))
        except (TypeError, ValueError):
            return None


class LLMDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    stops: list[LLMStop]
    notes: str = ""

    @field_validator("stops", mode="after")
    @classmethod
    def _non_empty(cls, v: list[LLMStop]) -> list[LLMStop]:
        if not v:
            raise ValueError("'stops' must contain at least one entry")
        return v


class LLMWhy(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    why: str = ""

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, v: Any) -> Any:
        if isinstance(v, dict) and "id" not in v:
            v = dict(v)
            for k in ("place_id", "placeId"):
                if k in v:
                    v["id"] = v[k]
                    break
        return v


class LLMFinal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    headline: str = ""
    summary: str = ""
    stops: list[LLMWhy] = Field(default_factory=list)
    trade_offs: list[str] = Field(default_factory=list)
    backup: str | None = None

    @field_validator("trade_offs", mode="before")
    @classmethod
    def _tl(cls, v: Any) -> Any:
        return _as_list(v) if isinstance(v, (str, list, tuple)) or v is None else []


# ------------------------------------------------------------ API result


class Totals(BaseModel):
    spend: int = 0
    travel_cost: int = 0
    total_cost: int = 0
    budget: int | None = None
    remaining: int | None = None
    start: int = 0
    end: int = 0
    minutes_available: int = 0
    minutes_used: int = 0
    travel_minutes: int = 0


class LLMStatus(BaseModel):
    enabled: bool = False  # a key is configured
    used: bool = False  # at least one model call succeeded for this plan
    model: str | None = None
    calls: int = 0
    tokens: int = 0
    notes: list[str] = Field(default_factory=list)


class PlanResult(BaseModel):
    status: Literal["ok", "needs_info", "fallback"] = "ok"
    mode: Literal["ai", "basic"] = "basic"
    headline: str = ""
    summary: str = ""
    city: str | None = None
    data_source: str = "curated"  # curated | openstreetmap | generic
    stops: list[ScheduledStop] = Field(default_factory=list)
    totals: Totals | None = None
    trade_offs: list[str] = Field(default_factory=list)
    backup: str | None = None
    cautions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    relaxed: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    message: str = ""  # plain-language note when there is no full plan
    preferences: Preferences | None = None
    llm: LLMStatus = Field(default_factory=LLMStatus)
    elapsed_ms: int = 0
