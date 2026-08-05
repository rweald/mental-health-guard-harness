# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Prompt fixtures shared across the suite.

The harness ships no prompt text, so every test that renders a prompt needs one on
disk. These fixtures write real temp files and let the real loader read them — no
monkeypatching of ``paper_prompts``. That way the normalizer, the placeholder check
and the explicit-path handling are exercised by ordinary tests rather than bypassed.

The fixture prompts are deliberately tiny and obviously synthetic. They are not
realistic safety prompts and should not be made into any: a fixture needs its
placeholder exactly once plus whatever structure the test under it asserts on, and
nothing more. Delimiters here are invented for the fixtures — tests that check
embedding assert against *these* markers, never against any real prompt's.
"""

import pytest

GUARD_FIXTURE = """FIXTURE GUARD PROMPT (test scaffolding, not a real safety prompt)
<<CONVO>>
{conversation}
<</CONVO>>
Reply with Safety: Safe or Safety: Unsafe."""

JUDGE_FIXTURE = """FIXTURE JUDGE PROMPT (test scaffolding, not a real safety prompt)
<<TURNS>>
{transcript}
<</TURNS>>
Answer as USER_TURN_N: label, once for each numbered turn above."""


@pytest.fixture
def guard_prompt_path(tmp_path):
    """A guard prompt file. Returns the path — callers pass it in explicitly."""
    path = tmp_path / "guard_fixture.txt"
    path.write_text(GUARD_FIXTURE, encoding="utf-8")
    return path


@pytest.fixture
def guard_fixture_text():
    """The text behind ``guard_prompt_path``, for tests asserting on exact rendering.
    A fixture rather than a cross-module import — ``tests`` is not a package."""
    return GUARD_FIXTURE


@pytest.fixture
def judge_prompt_path(tmp_path):
    """A judge prompt file. Returns the path — callers pass it in explicitly."""
    path = tmp_path / "judge_fixture.txt"
    path.write_text(JUDGE_FIXTURE, encoding="utf-8")
    return path
