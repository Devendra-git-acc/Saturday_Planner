"""LLM access. One narrow interface (`complete_json`) so the agent never touches the SDK directly.

Design notes
- We ask for JSON, but never trust it: output is parsed leniently, validated with Pydantic,
  retried once with the error fed back, and finally handed to a deterministic fallback.
- Model/parameter quirks differ across OpenAI generations (reasoning_effort, temperature,
  max_tokens vs max_completion_tokens, JSON mode). If the API rejects a parameter we drop or swap
  it and retry, so a model rename in OPENAI_MODEL does not take the app down.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings
from .trace import ToolInfo
from .util import short

log = logging.getLogger("saturday.llm")

M = TypeVar("M", bound=BaseModel)


class LLMUnavailable(Exception):
    """The model cannot be used right now (network, auth, quota, unknown model...)."""


class LLMBadOutput(Exception):
    """The model answered, but not with usable JSON."""

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


@dataclass
class LLMResult:
    data: dict
    tokens: int = 0
    ms: int = 0
    model: str = ""
    raw: str = ""


class LLM(Protocol):
    name: str
    model: str

    def complete_json(
        self, task: str, messages: list[dict], *, effort: str | None = None, max_tokens: int | None = None
    ) -> LLMResult: ...


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def extract_json(text: str) -> dict:
    """Parse a JSON object out of model text, tolerating code fences and stray prose."""
    if not text or not text.strip():
        raise LLMBadOutput("empty response", text or "")
    s = _FENCE.sub("", text.strip()).strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i == -1 or j <= i:
            raise LLMBadOutput("response is not JSON", text)
        try:
            data = json.loads(s[i : j + 1])
        except json.JSONDecodeError as e:
            raise LLMBadOutput(f"invalid JSON ({e.msg})", text) from e
    if not isinstance(data, dict):
        raise LLMBadOutput("JSON root must be an object", text)
    return data


DEFAULT_BASE_URL = "https://api.openai.com/v1"
_REASONING_MODEL = re.compile(r"^(gpt-[5-9]|o\d)", re.I)


class OpenAILLM:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        mode: str = "chat",
        reasoning: str = "auto",
        timeout: float = 25.0,
        max_tokens: int = 2500,
        http_client: Any = None,
        max_retries: int = 1,
    ) -> None:
        import openai  # imported lazily so the rest of the app works without the SDK

        self._o = openai
        self.model = model
        self.mode = mode if mode in {"chat", "responses"} else "chat"
        self.reasoning = reasoning or "auto"
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(
            # Always pass an address: with base_url=None the SDK reads OPENAI_BASE_URL itself, and an empty value
            # (as in a .env copied from .env.example) becomes a blank address that fails every call instantly.
            api_key=api_key, base_url=base_url or DEFAULT_BASE_URL, timeout=timeout, max_retries=max_retries,
            http_client=http_client,
        )
        # learned at runtime when the API rejects something
        self._dropped: set[str] = set()
        self._token_param = "max_completion_tokens"
        self._effort_map: dict[str, str] = {}
        # circuit breaker: after repeated infrastructure failures stop waiting on a dead API
        self._fails = 0
        self._cooldown_until = 0.0

    # ------------------------------------------------------------ public
    def complete_json(
        self, task: str, messages: list[dict], *, effort: str | None = None, max_tokens: int | None = None
    ) -> LLMResult:
        if time.monotonic() < self._cooldown_until:
            raise LLMUnavailable("AI temporarily paused after repeated errors; using basic mode")
        t0 = time.perf_counter()
        try:
            if self.mode == "responses":
                text, tokens = self._call_responses(messages, effort, max_tokens)
            else:
                text, tokens = self._call_chat(messages, effort, max_tokens)
        except LLMUnavailable as e:
            log.warning("model call failed (%s): %s; cause: %s", task, e, short(e.__cause__) if e.__cause__ else "none")
            self._fails += 1
            if self._fails >= 2:
                self._cooldown_until = time.monotonic() + 45.0
            raise
        self._fails = 0
        data = extract_json(text)
        return LLMResult(data=data, tokens=tokens, ms=int((time.perf_counter() - t0) * 1000), model=self.model, raw=text)

    # ----------------------------------------------------------- helpers
    def _effort(self, hint: str | None) -> str | None:
        policy = (self.reasoning or "auto").lower()
        if policy == "off" or "reasoning_effort" in self._dropped:
            return None
        if policy != "auto":
            return self._effort_map.get(policy, policy)
        if not _REASONING_MODEL.match(self.model) or not hint:
            return None
        return self._effort_map.get(hint, hint)

    def _call_chat(self, messages: list[dict], effort: str | None, max_tokens: int | None) -> tuple[str, int]:
        o = self._o
        params: dict[str, Any] = {"model": self.model, "messages": messages}
        if "response_format" not in self._dropped:
            params["response_format"] = {"type": "json_object"}
        params[self._token_param] = max_tokens or self.max_tokens
        eff = self._effort(effort)
        if eff:
            params["reasoning_effort"] = eff

        resp = None
        for _ in range(4):
            try:
                resp = self.client.chat.completions.create(**params)
                break
            except o.BadRequestError as e:
                if not self._heal(params, str(e)):
                    raise LLMUnavailable(f"OpenAI rejected the request: {short(e)}") from e
            except o.AuthenticationError as e:
                raise LLMUnavailable("OpenAI rejected the API key (401)") from e
            except o.PermissionDeniedError as e:
                raise LLMUnavailable("OpenAI denied access to this model or account (403)") from e
            except o.NotFoundError as e:
                raise LLMUnavailable(f"model '{self.model}' was not found for this key (404)") from e
            except o.RateLimitError as e:
                raise LLMUnavailable("OpenAI rate limit or quota reached (429)") from e
            except (o.APITimeoutError, o.APIConnectionError) as e:
                raise LLMUnavailable("could not reach OpenAI (timeout or connection error)") from e
            except o.APIStatusError as e:
                raise LLMUnavailable(f"OpenAI returned HTTP {e.status_code}") from e
        if resp is None:
            raise LLMUnavailable("OpenAI kept rejecting request parameters")

        choice = resp.choices[0]
        text = choice.message.content or ""
        if not text.strip() and getattr(choice, "finish_reason", None) == "length":
            raise LLMBadOutput("output was cut off before any JSON (token limit reached)")
        tokens = getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
        return text, int(tokens)

    def _call_responses(self, messages: list[dict], effort: str | None, max_tokens: int | None) -> tuple[str, int]:
        o = self._o
        params: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "max_output_tokens": max_tokens or self.max_tokens,
        }
        if "response_format" not in self._dropped:
            params["text"] = {"format": {"type": "json_object"}}
        eff = self._effort(effort)
        if eff:
            params["reasoning"] = {"effort": eff}

        resp = None
        for _ in range(4):
            try:
                resp = self.client.responses.create(**params)
                break
            except o.BadRequestError as e:
                m = str(e).lower()
                if "reasoning" in params and "reasoning" in m:
                    params.pop("reasoning")
                    self._dropped.add("reasoning_effort")
                elif "text" in params and ("text" in m or "format" in m or "json" in m):
                    params.pop("text")
                    self._dropped.add("response_format")
                else:
                    raise LLMUnavailable(f"OpenAI rejected the request: {short(e)}") from e
            except o.AuthenticationError as e:
                raise LLMUnavailable("OpenAI rejected the API key (401)") from e
            except o.PermissionDeniedError as e:
                raise LLMUnavailable("OpenAI denied access to this model or account (403)") from e
            except o.NotFoundError as e:
                raise LLMUnavailable(f"model '{self.model}' was not found for this key (404)") from e
            except o.RateLimitError as e:
                raise LLMUnavailable("OpenAI rate limit or quota reached (429)") from e
            except (o.APITimeoutError, o.APIConnectionError) as e:
                raise LLMUnavailable("could not reach OpenAI (timeout or connection error)") from e
            except o.APIStatusError as e:
                raise LLMUnavailable(f"OpenAI returned HTTP {e.status_code}") from e
        if resp is None:
            raise LLMUnavailable("OpenAI kept rejecting request parameters")
        text = getattr(resp, "output_text", "") or ""
        tokens = getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
        return text, int(tokens)

    def _heal(self, params: dict, msg: str) -> bool:
        """The API said 400. If it names a parameter we added, drop/swap it and try again."""
        m = msg.lower()
        if "reasoning_effort" in params and "reasoning" in m:
            value_problem = any(t in m for t in ("unsupported value", "invalid value", "must be one of", "supported values"))
            if value_problem and params["reasoning_effort"] == "none":
                params["reasoning_effort"] = "low"
                self._effort_map["none"] = "low"
                return True
            params.pop("reasoning_effort")
            self._dropped.add("reasoning_effort")
            return True
        if "response_format" in params and any(t in m for t in ("response_format", "json_object", "json mode")):
            params.pop("response_format")
            self._dropped.add("response_format")
            return True
        if "temperature" in params and "temperature" in m:
            params.pop("temperature")
            return True
        if "max_completion_tokens" in params and "max_completion_tokens" in m:
            params["max_tokens"] = params.pop("max_completion_tokens")
            self._token_param = "max_tokens"
            return True
        if "max_tokens" in params and "max_tokens" in m:
            params["max_completion_tokens"] = params.pop("max_tokens")
            self._token_param = "max_completion_tokens"
            return True
        return False


def build_llm(settings: Settings) -> LLM | None:
    """None means 'run in basic mode' (no key, or the SDK is unavailable)."""
    if not settings.openai_api_key:
        return None
    try:
        return OpenAILLM(
            settings.openai_api_key,
            settings.openai_model,
            base_url=settings.openai_base_url,
            mode=settings.openai_api_mode,
            reasoning=settings.reasoning_effort,
            timeout=settings.llm_timeout_s,
            max_tokens=settings.llm_max_output_tokens,
        )
    except Exception:  # noqa: BLE001
        log.exception("Could not initialise the OpenAI client; running in basic mode")
        return None


def call_validated(
    llm: LLM,
    task: str,
    messages: list[dict],
    model_cls: type[M],
    *,
    effort: str | None,
    info: ToolInfo,
    retries: int = 1,
) -> M:
    """Call the model, validate against `model_cls`; on bad output retry once with the error.

    Raises LLMBadOutput after the final attempt; LLMUnavailable passes straight through.
    """
    msgs = list(messages)
    last: Exception | None = None
    for attempt in range(retries + 1):
        raw = ""
        try:
            res = llm.complete_json(task, msgs, effort=effort)
            info.llm_calls += 1
            info.tokens += res.tokens
            raw = res.raw or json.dumps(res.data, ensure_ascii=False)
            return model_cls.model_validate(res.data)
        except LLMBadOutput as e:
            info.llm_calls += 1
            last, raw = e, e.raw
        except ValidationError as e:
            last = e
        if attempt < retries:
            info.retries += 1
            problem = short(last, 300) if last else "unknown problem"
            msgs = msgs + [
                {"role": "assistant", "content": (raw or "")[:1500] or "(empty)"},
                {
                    "role": "user",
                    "content": (
                        f"Your previous reply could not be used: {problem}. "
                        "Reply again with ONLY one valid JSON object that follows the schema exactly."
                    ),
                },
            ]
    raise LLMBadOutput(short(last, 200) if last else "invalid output")