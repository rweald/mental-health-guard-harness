# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Model adapter layer: one ``Model`` protocol, thin concrete adapters over the
official SDKs. Every pipeline step takes a ``Model``; an "arm" config names which
adapter fills each role (PLM / CLM / judge), so swapping models is a config change,
not a code change.

OpenRouter's API is OpenAI-compatible, so a single adapter covers GLM-4.6, GLM-5.2,
and DeepSeek V4 Flash via a base-URL + model-id swap. Kept deliberately tiny (~protocol
+ one concrete class): we want raw provider responses in hand, not a framework.

Decision models get a second, parallel protocol (``DecisionModel``) at the bottom of
the module: they answer typed questions with probabilities instead of generating text.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

Message = dict[str, str]  # {"role": ..., "content": ...}


@dataclass
class Completion:
    """One model response. ``raw`` keeps the provider object (for logprobs/usage)."""

    text: str
    raw: Any = None


@runtime_checkable
class Model(Protocol):
    """Anything the pipeline can call. One sample per call — callers that want N
    samples (majority vote) loop, so a provider that ignores ``n`` can't silently
    collapse the vote."""

    def complete(self, messages: list[Message], **opts: Any) -> Completion: ...


@dataclass
class OpenRouterModel:
    """Adapter over the OpenAI SDK pointed at OpenRouter. ``model`` is the OpenRouter
    id (e.g. ``z-ai/glm-4.6``). ``default_opts`` carries per-model knobs (temperature,
    reasoning effort) that an arm config sets once; per-call ``opts`` override them."""

    model: str
    api_key: str | None = None  # defaults to $OPENROUTER_API_KEY
    base_url: str = "https://openrouter.ai/api/v1"
    default_opts: dict[str, Any] = field(default_factory=dict)
    # A stalled OpenRouter response otherwise blocks on the socket read indefinitely
    # (the SDK default is ~10 min). Bound every call so a hang raises APITimeoutError,
    # which the SDK retries with backoff, then surfaces loudly. Raise `timeout` for
    # max-reasoning judges (GLM-5.2 Max, DeepSeek xhigh) if legit calls run long.
    timeout: float = 120.0
    max_retries: int = 2
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from openai import OpenAI

        key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set (and no api_key passed)")
        self._client = OpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def complete(self, messages: list[Message], **opts: Any) -> Completion:
        params = {**self.default_opts, **opts}
        resp = self._client.chat.completions.create(
            model=self.model, messages=list(messages), **params
        )
        return Completion(text=_extract_text(resp), raw=resp)


def _extract_text(resp: Any) -> str:
    """Pull the assistant text out of a chat-completions response, tolerating the shapes
    providers actually return: no choices at all (hard content-filter), a ``None``
    content, or a list of content-blocks (some reasoning models). Always returns a
    string, so callers never crash on ``.strip()`` and an empty result flows into the
    caller's own empty-output handling rather than dropping the whole call opaquely."""
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ""
    content = getattr(choices[0].message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content-blocks: concatenate their text
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
            else:
                parts.append(getattr(p, "text", "") or "")
        return "".join(parts)
    return str(content)


# --- Decision models -----------------------------------------------------------------
#
# Decision models (Jev, Clef, OpenAI Decisions, ...) do not generate text: they take a
# ``state`` plus a map of typed questions and return a probability per allowed answer.
# That is a different shape from ``Model.complete``, so it gets its own protocol rather
# than being bent into a text completion. OpenRouter serves all of them behind one
# endpoint (``POST /api/alpha/decisions``), so one adapter covers every arm.


@dataclass
class Decision:
    """One decision-model response. ``answers`` is keyed by question name; ``raw`` is
    the whole parsed response body (resolved model version, provider, usage/cost)."""

    answers: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DecisionModel(Protocol):
    """Anything that answers typed questions about a state."""

    def decide(self, state: Any, questions: dict[str, Any]) -> Decision: ...


class DecisionAPIError(RuntimeError):
    """A decisions request that failed after retries. ``status`` is the HTTP status, or
    ``None`` for a transport failure or an unusable 200 body. ``fatal`` errors (bad key,
    no credits, forbidden) will fail every other row the same way, so callers should
    stop the run rather than record them per row."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status

    @property
    def fatal(self) -> bool:
        return self.status in _FATAL_STATUSES


# OpenRouter's documented transient statuses (rate limit, upstream failure/overload,
# edge timeout) plus 408 request timeout. Others won't fix themselves on retry: most
# (400 bad request, 413 too large) are about the row, and surface for that row only;
# the fatal ones are about the account, and stop the run.
_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 524, 529})
_FATAL_STATUSES = frozenset({401, 402, 403})
_MAX_RETRY_AFTER_S = 60.0


@dataclass
class OpenRouterDecisionModel:
    """Adapter over OpenRouter's Decisions endpoint. ``model`` is the OpenRouter id
    (e.g. ``typesafe/jev-1.14``). ``provider`` is passed through as OpenRouter's
    provider-routing object — pin it so a run can't drift between hosts that serve
    different quantizations or context limits of the same model."""

    model: str
    provider: dict[str, Any] | None = None
    api_key: str | None = None  # defaults to $OPENROUTER_API_KEY
    base_url: str = "https://openrouter.ai/api/alpha"
    timeout: float = 60.0
    max_retries: int = 3
    backoff_s: float = 2.0  # doubles per retry; a longer Retry-After wins
    transport: Any = None  # httpx transport override, for tests
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import httpx

        key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set (and no api_key passed)")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=self.timeout,
            transport=self.transport,
        )

    def decide(self, state: Any, questions: dict[str, Any]) -> Decision:
        import httpx

        body: dict[str, Any] = {"model": self.model, "state": state, "questions": questions}
        if self.provider:
            body["provider"] = self.provider

        last_error: DecisionAPIError | None = None
        wait_s = 0.0
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(max(wait_s, self.backoff_s * 2 ** (attempt - 1)))
            wait_s = 0.0
            try:
                resp = self._client.post("/decisions", json=body)
            except httpx.TransportError as e:
                last_error = DecisionAPIError(f"transport error: {e!r}")
                continue
            if resp.status_code == 200:
                data, problem = _parse_decision_body(resp)
                if problem is None:
                    return Decision(answers=data["answers"], raw=data)
                # A 200 carrying an error or garbage is an upstream hiccup, not an
                # answer: never cache it as one, and give it the transient treatment.
                last_error = DecisionAPIError(f"HTTP 200 but {problem}")
                continue
            last_error = DecisionAPIError(
                f"HTTP {resp.status_code}: {_error_message(resp)}", status=resp.status_code
            )
            if resp.status_code not in _RETRYABLE_STATUSES:
                break
            wait_s = _retry_after_s(resp)
        raise last_error


def _parse_decision_body(resp: Any) -> tuple[dict | None, str | None]:
    """The parsed body if it is a usable decision, else why not."""
    try:
        data = resp.json()
    except ValueError:
        return None, "the body is not JSON"
    if not isinstance(data, dict):
        return None, "the body is not a JSON object"
    if "error" in data:
        return None, f"the body is an error: {_error_message(resp)}"
    if not isinstance(data.get("answers"), dict):
        return None, "the body has no answers object"
    return data, None


def _retry_after_s(resp: Any) -> float:
    """Seconds from a numeric ``Retry-After`` header, capped; 0 if absent or a date."""
    try:
        return min(max(float(resp.headers.get("retry-after", 0)), 0.0), _MAX_RETRY_AFTER_S)
    except ValueError:
        return 0.0


def _error_message(resp: Any) -> str:
    """OpenRouter's ``{"error": {"message": ...}}``, else the raw body (truncated)."""
    try:
        return str(resp.json()["error"]["message"])
    except Exception:
        return resp.text[:500]
