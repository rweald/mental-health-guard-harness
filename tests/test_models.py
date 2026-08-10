# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Model adapter layer: protocol conformance + OpenRouter adapter wiring (mocked SDK)."""

import sys
import types

import pytest

from mh_guard_harness.models import (
    Completion,
    Model,
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
