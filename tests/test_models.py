# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Model adapter layer: protocol conformance + OpenRouter adapter wiring (mocked SDK /
mocked HTTP transport)."""

import json
import sys
import types

import httpx
import pytest

from mh_guard_harness.models import (
    Completion,
    Decision,
    DecisionAPIError,
    DecisionModel,
    Model,
    OpenRouterDecisionModel,
    OpenRouterModel,
    _extract_text,
)


def _resp(content):
    msg = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_extract_text_plain_string():
    assert _extract_text(_resp("hello")) == "hello"


def test_extract_text_none_content_is_empty():
    # a reasoning-only / filtered response with null content must not crash callers
    assert _extract_text(_resp(None)) == ""


def test_extract_text_empty_choices_is_empty():
    # hard content-filter can return no choices at all; don't IndexError the whole call
    assert _extract_text(types.SimpleNamespace(choices=[])) == ""


def test_extract_text_joins_content_blocks():
    blocks = [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]
    assert _extract_text(_resp(blocks)) == "part one part two"


class ScriptedModel:
    """A Model that returns queued texts — the test double the judge labeler uses."""

    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = []

    def complete(self, messages, **opts):
        self.calls.append({"messages": messages, "opts": opts})
        return Completion(text=self._texts.pop(0))


def test_scripted_model_satisfies_protocol():
    assert isinstance(ScriptedModel(["x"]), Model)


def test_completion_carries_text_and_raw():
    c = Completion(text="hi", raw={"k": 1})
    assert c.text == "hi" and c.raw == {"k": 1}


def _install_fake_openai(monkeypatch, capture):
    """Install a fake ``openai`` module whose client records the create() call."""
    mod = types.ModuleType("openai")

    class FakeClient:
        def __init__(self, api_key, base_url, timeout=None, max_retries=None):
            capture["init"] = {
                "api_key": api_key, "base_url": base_url,
                "timeout": timeout, "max_retries": max_retries,
            }
            chat = types.SimpleNamespace()
            chat.completions = types.SimpleNamespace(create=self._create)
            self.chat = chat

        def _create(self, **kwargs):
            capture["create"] = kwargs
            msg = types.SimpleNamespace(content="Safety: Safe")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    mod.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", mod)
    return capture


def test_openrouter_uses_env_key_and_base_url(monkeypatch):
    cap = {}
    _install_fake_openai(monkeypatch, cap)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    m = OpenRouterModel(model="z-ai/glm-4.6", default_opts={"temperature": 0.7})
    out = m.complete([{"role": "user", "content": "hi"}], n=1)
    assert out.text == "Safety: Safe"
    assert cap["init"]["api_key"] == "sk-test"
    assert cap["init"]["base_url"] == "https://openrouter.ai/api/v1"
    # timeout + retries are set so a stalled call can't hang indefinitely
    assert cap["init"]["timeout"] == 120.0
    assert cap["init"]["max_retries"] == 2
    # default_opts merge + per-call opts both reach the SDK
    assert cap["create"]["model"] == "z-ai/glm-4.6"
    assert cap["create"]["temperature"] == 0.7
    assert cap["create"]["n"] == 1


def test_per_call_opts_override_defaults(monkeypatch):
    cap = {}
    _install_fake_openai(monkeypatch, cap)
    m = OpenRouterModel(model="x", api_key="k", default_opts={"temperature": 0.0})
    m.complete([{"role": "user", "content": "hi"}], temperature=1.0)
    assert cap["create"]["temperature"] == 1.0


def test_missing_key_raises(monkeypatch):
    _install_fake_openai(monkeypatch, {})
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterModel(model="x")


# --- Decision models -----------------------------------------------------------------


def _decision_model(handler, **kw):
    return OpenRouterDecisionModel(
        model="vendor/decider", api_key="sk-test", backoff_s=0.0,
        transport=httpx.MockTransport(handler), **kw,
    )


def test_decision_model_posts_state_questions_and_provider():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "vendor/decider-1", "answers": {"q": {}}})

    m = _decision_model(handler, provider={"only": ["vendor"]})
    out = m.decide({"s": 1}, {"q": {"type": "noul"}})
    assert seen["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"] == {
        "model": "vendor/decider", "state": {"s": 1},
        "questions": {"q": {"type": "noul"}}, "provider": {"only": ["vendor"]},
    }
    assert out.answers == {"q": {}}
    assert out.raw["model"] == "vendor/decider-1"


def test_decision_model_omits_provider_when_unset():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answers": {}})

    _decision_model(handler).decide("s", {})
    assert "provider" not in seen["body"]


def test_decision_model_retries_transient_then_succeeds():
    statuses = iter([429, 503, 200])

    def handler(request):
        status = next(statuses)
        return httpx.Response(status, json={"answers": {"ok": True}} if status == 200 else {})

    assert _decision_model(handler).decide("s", {}).answers == {"ok": True}


def test_decision_model_does_not_retry_bad_request():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"code": 400, "message": "bad schema"}})

    with pytest.raises(DecisionAPIError, match="bad schema") as e:
        _decision_model(handler).decide("s", {})
    assert e.value.status == 400
    assert len(calls) == 1


def test_decision_model_gives_up_after_max_retries():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(502, text="upstream down")

    with pytest.raises(DecisionAPIError) as e:
        _decision_model(handler, max_retries=2).decide("s", {})
    assert e.value.status == 502
    assert len(calls) == 3


def test_decision_model_transport_error_is_api_error():
    def handler(request):
        raise httpx.ConnectError("reset")

    with pytest.raises(DecisionAPIError, match="transport") as e:
        _decision_model(handler, max_retries=0).decide("s", {})
    assert e.value.status is None


def test_decision_model_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterDecisionModel(model="x")


def test_scripted_decider_satisfies_protocol():
    class D:
        def decide(self, state, questions):
            return Decision(answers={})

    assert isinstance(D(), DecisionModel)


def test_decision_model_retries_408():
    statuses = iter([408, 200])

    def handler(request):
        status = next(statuses)
        return httpx.Response(status, json={"answers": {}} if status == 200 else {})

    assert _decision_model(handler).decide("s", {}).answers == {}


def test_decision_model_honours_retry_after(monkeypatch):
    slept = []
    monkeypatch.setattr("mh_guard_harness.models.time.sleep", slept.append)
    statuses = iter([429, 429, 200])

    def handler(request):
        status = next(statuses)
        if status == 200:
            return httpx.Response(200, json={"answers": {}})
        return httpx.Response(429, headers={"Retry-After": "7" if not slept else "999"})

    _decision_model(handler).decide("s", {})
    assert slept == [7.0, 60.0]  # honoured, and capped


@pytest.mark.parametrize("status", [401, 402, 403])
def test_account_errors_are_fatal_and_not_retried(status):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(status, json={"error": {"code": status, "message": "no"}})

    with pytest.raises(DecisionAPIError) as e:
        _decision_model(handler).decide("s", {})
    assert e.value.fatal and len(calls) == 1


def test_row_errors_are_not_fatal():
    def handler(request):
        return httpx.Response(400, json={"error": {"code": 400, "message": "bad"}})

    with pytest.raises(DecisionAPIError) as e:
        _decision_model(handler).decide("s", {})
    assert not e.value.fatal


@pytest.mark.parametrize("response, why", [
    (httpx.Response(200, json={"error": {"code": 502, "message": "upstream"}}), "error"),
    (httpx.Response(200, text="<html>gateway</html>"), "not JSON"),
    (httpx.Response(200, json=[1, 2]), "not a JSON object"),
    (httpx.Response(200, json={"model": "m"}), "no answers"),
])
def test_unusable_200_is_retried_then_raised_not_cached(response, why):
    calls = []

    def handler(request):
        calls.append(1)
        return response

    with pytest.raises(DecisionAPIError, match=why):
        _decision_model(handler, max_retries=1).decide("s", {})
    assert len(calls) == 2
