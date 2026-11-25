"""The OpenAI client against a mocked HTTP transport (the real API is not reachable from CI/sandbox).

Verifies the request shape we send and that the client heals itself when a model rejects a parameter.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.llm import LLMBadOutput, LLMUnavailable, OpenAILLM, call_validated, extract_json
from app.schemas import LLMDraft
from app.trace import ToolInfo


def chat_ok(content, finish="stop"):
    return {
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def err(status, message, param=None):
    return httpx.Response(status, json={"error": {"message": message, "type": "invalid_request_error", "param": param, "code": None}})


def make(handler, model="gpt-6-luna", **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAILLM("sk-test", model, http_client=client, max_retries=0, **kw)


MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def test_request_shape_for_a_reasoning_model():
    seen = []

    def h(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=chat_ok('{"a": 1}'))

    res = make(h).complete_json("t", MSGS, effort="low")
    body = seen[0]
    assert res.data == {"a": 1} and res.tokens == 12
    assert body["model"] == "gpt-6-luna"
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_completion_tokens"] == 2500
    assert body["reasoning_effort"] == "low"
    assert "temperature" not in body and "max_tokens" not in body


def test_no_reasoning_effort_for_non_reasoning_models():
    seen = []

    def h(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=chat_ok("{}"))

    make(h, model="some-chat-model").complete_json("t", MSGS, effort="low")
    assert "reasoning_effort" not in seen[0]


def test_heals_when_reasoning_effort_is_rejected():
    seen = []

    def h(req):
        body = json.loads(req.content)
        seen.append(body)
        if "reasoning_effort" in body:
            return err(400, "Unsupported parameter: 'reasoning_effort' is not supported with this model.", "reasoning_effort")
        return httpx.Response(200, json=chat_ok('{"ok": true}'))

    llm = make(h)
    assert llm.complete_json("t", MSGS, effort="low").data == {"ok": True}
    assert len(seen) == 2
    llm.complete_json("t", MSGS, effort="low")
    assert "reasoning_effort" not in seen[-1] and len(seen) == 3  # remembered, no second failure


def test_heals_unsupported_effort_value():
    def h(req):
        body = json.loads(req.content)
        if body.get("reasoning_effort") == "low":
            return err(400, "Unsupported value: 'low' for reasoning_effort. Supported values are: 'medium'.", "reasoning_effort")
        return httpx.Response(200, json=chat_ok("{}"))

    llm = make(h)
    try:
        llm.complete_json("t", MSGS, effort="low")
    except LLMUnavailable:
        pass  # allowed: we only require no crash and no infinite loop


def test_heals_json_mode_and_token_param():
    seen = []

    def h(req):
        body = json.loads(req.content)
        seen.append(body)
        if "response_format" in body:
            return err(400, "response_format json_object is not supported for this model", "response_format")
        if "max_completion_tokens" in body:
            return err(400, "Unsupported parameter: 'max_completion_tokens'. Use 'max_tokens' instead.", "max_completion_tokens")
        return httpx.Response(200, json=chat_ok('{"x": 2}'))

    res = make(h).complete_json("t", MSGS)
    assert res.data == {"x": 2}
    assert "max_tokens" in seen[-1] and "response_format" not in seen[-1]


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500])
def test_http_failures_become_llm_unavailable(status):
    llm = make(lambda r: err(status, "nope"))
    with pytest.raises(LLMUnavailable):
        llm.complete_json("t", MSGS)


def test_connection_errors_become_llm_unavailable():
    def h(req):
        raise httpx.ConnectError("boom")

    with pytest.raises(LLMUnavailable):
        make(h).complete_json("t", MSGS)


def test_truncated_reply_is_bad_output_not_a_crash():
    llm = make(lambda r: httpx.Response(200, json=chat_ok("", finish="length")))
    with pytest.raises(LLMBadOutput):
        llm.complete_json("t", MSGS)


def test_circuit_breaker_stops_hammering_a_dead_api():
    calls = []

    def h(req):
        calls.append(1)
        return err(500, "down")

    llm = make(h)
    for _ in range(2):
        with pytest.raises(LLMUnavailable):
            llm.complete_json("t", MSGS)
    n = len(calls)
    with pytest.raises(LLMUnavailable, match="paused"):
        llm.complete_json("t", MSGS)
    assert len(calls) == n  # no HTTP call while the breaker is open


def test_responses_api_mode():
    seen = []

    def h(req):
        seen.append((req.url.path, json.loads(req.content)))
        return httpx.Response(200, json={
            "id": "r", "object": "response", "created_at": 0, "model": "m", "status": "completed", "parallel_tool_calls": False,
            "tool_choice": "auto", "tools": [],
            "output": [{"type": "message", "id": "m1", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": '{"r": 1}', "annotations": []}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
        })

    res = make(h, mode="responses").complete_json("t", MSGS, effort="low")
    assert res.data == {"r": 1}
    path, body = seen[0]
    assert path.endswith("/responses")
    assert body["text"] == {"format": {"type": "json_object"}} and body["reasoning"] == {"effort": "low"}


# ---- extract_json / call_validated


@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Sure! Here you go:\n{"a": {"b": [1, 2]}}\nHope that helps', {"a": {"b": [1, 2]}}),
])
def test_extract_json_accepts_messy_wrappers(raw, expected):
    assert extract_json(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "no json here", "[1, 2, 3]", '{"a": ', "null"])
def test_extract_json_rejects_non_objects(raw):
    with pytest.raises(LLMBadOutput):
        extract_json(raw)


def test_call_validated_retries_once_with_the_error_fed_back():
    replies = iter([{"stops": []}, {"stops": [{"id": "a", "duration_min": 30}]}])
    seen = []

    class L:
        name, model = "x", "x"

        def complete_json(self, task, messages, **kw):
            seen.append(messages)
            from app.llm import LLMResult

            return LLMResult(data=next(replies), tokens=1)

    info = ToolInfo()
    out = call_validated(L(), "draft", MSGS, LLMDraft, effort=None, info=info)
    assert out.stops[0].id == "a" and info.retries == 1 and info.llm_calls == 2
    assert "could not be used" in seen[1][-1]["content"]


def test_call_validated_gives_up_after_the_retry():
    class L:
        name, model = "x", "x"

        def complete_json(self, task, messages, **kw):
            from app.llm import LLMResult

            return LLMResult(data={"stops": []}, tokens=1)

    with pytest.raises(LLMBadOutput):
        call_validated(L(), "draft", MSGS, LLMDraft, effort=None, info=ToolInfo())


def test_an_empty_openai_base_url_in_the_environment_does_not_break_the_client(monkeypatch):
    """A .env copied from .env.example has `OPENAI_BASE_URL=`; the SDK used to treat that as a blank address."""
    from app.llm import OpenAILLM

    monkeypatch.setenv("OPENAI_BASE_URL", "")
    assert str(OpenAILLM("sk-test", "gpt-6-luna").client.base_url) == "https://api.openai.com/v1/"
    assert str(OpenAILLM("sk-test", "m", base_url="https://example.test/v1").client.base_url) == "https://example.test/v1/"