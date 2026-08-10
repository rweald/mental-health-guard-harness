# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Model adapter layer: one ``Model`` protocol, thin concrete adapters over the
official SDKs. Every pipeline step takes a ``Model``; an "arm" config names which
adapter fills each role (PLM / CLM / judge), so swapping models is a config change,
not a code change.

OpenRouter's API is OpenAI-compatible, so a single adapter covers GLM-4.6, GLM-5.2,
and DeepSeek V4 Flash via a base-URL + model-id swap. Kept deliberately tiny (~protocol
+ one concrete class): we want raw provider responses in hand, not a framework.
"""

from __future__ import annotations

import os
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
