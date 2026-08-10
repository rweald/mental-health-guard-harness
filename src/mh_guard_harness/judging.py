# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""§7.2 judge labeler: label each USER turn safe / self-harm / threats-to-others via
N-sample majority vote. Two context modes isolate a single variable — whether the
judge sees future turns:

  - MODE_FULL ("full"): the paper's method. All user turns go in the numbered
    transcript, so each turn is labeled with full-conversation context (incl. later
    turns) — higher-quality supervision, §5.
  - MODE_TURN ("turn"): our variant. For target turn t the transcript holds only user
    turns 1..t, so the judge sees past context only — matching how the testset gold,
    the deployed classifier, and our fine-tuned student all operate.

Same judge prompt both ways; only the transcript length differs. The transcript is USER
turns only (per §5/§7.2) — assistant/clinician turns are not shown to the judge.

The prompt itself is not defined here: it is loaded at call time from a caller-supplied
file (see ``paper_prompts``), so the harness ships no prompt text of its own.
"""

from __future__ import annotations

import re
from collections import Counter

from . import paper_prompts
from .conversations import Conversation
from .models import Model

MODE_FULL = "full"
MODE_TURN = "turn"

# The three labels the judge emits (§7.2). Note these differ from the testset gold's
# spelling: the judge says `unsafe_threats_to_others`, the gold says
# `unsafe_harm_to_others` — the agreement step maps both to a common space.
LABELS = ("safe", "unsafe_self_harm_risk", "unsafe_threats_to_others")

# Severity order for majority-vote tie-breaks (safety-conservative: unsafe wins ties).
_SEVERITY = {"safe": 0, "unsafe_self_harm_risk": 1, "unsafe_threats_to_others": 2}

# Lenient normalization of what the model actually writes -> canonical judge labels.
_ALIASES = {
    "safe": "safe",
    "unsafe_self_harm_risk": "unsafe_self_harm_risk",
    "self_harm": "unsafe_self_harm_risk",
    "unsafe_threats_to_others": "unsafe_threats_to_others",
    "unsafe_harm_to_others": "unsafe_threats_to_others",
    "threats_to_others": "unsafe_threats_to_others",
}

_LINE_RE = re.compile(r"USER_TURN_(\d+)\s*:\s*([A-Za-z_]+)", re.IGNORECASE)


def _user_turns(conversation: Conversation) -> list[str]:
    return [m["content"] for m in conversation.messages if m["role"] == "user"]


def render_transcript(user_turns: list[str]) -> str:
    """Numbered user-turn transcript, matching the USER_TURN_N output keys. Newlines
    inside a turn are flattened to spaces so one turn stays on one line — otherwise an
    embedded newline would split a turn across lines and misalign the numbering the judge
    reads. (No-op on the testset, whose turns are already single-line.)"""
    return "\n".join(
        f"USER_TURN_{i}: {t.replace(chr(10), ' ').replace(chr(13), ' ')}"
        for i, t in enumerate(user_turns, start=1)
    )


def build_judge_prompt(
    user_turns: list[str],
    *,
    prompt_path,
    reproduction_mode: bool = False,
) -> str:
    """Render the judge prompt around the numbered transcript.

    The prompt is loaded from ``prompt_path`` on every call — the file is tiny and
    reloading it keeps the loader stateless and cache-free."""
    return paper_prompts.load_and_render(
        "judge",
        render_transcript(user_turns),
        path=prompt_path,
        mode=paper_prompts.mode_for(reproduction_mode),
    )


def parse_labels(text: str) -> dict[int, str]:
    """Parse ``USER_TURN_N: label`` lines -> {turn_number: canonical_label}. Unknown
    label spellings are dropped (treated as no-vote for that turn)."""
    out: dict[int, str] = {}
    for m in _LINE_RE.finditer(text):
        n = int(m.group(1))
        label = _ALIASES.get(m.group(2).strip().lower())
        if label is not None:
            out[n] = label
    return out


def majority_vote(labels: list[str]) -> str | None:
    """Plurality label; ties broken toward the more severe (safety-conservative).
    Returns None if there are no valid labels."""
    if not labels:
        return None
    counts = Counter(labels)
    top = max(counts.values())
    tied = [lab for lab, c in counts.items() if c == top]
    return max(tied, key=lambda lab: _SEVERITY.get(lab, 0))


def _sample_labelings(
    model: Model,
    user_turns: list[str],
    n_samples: int,
    opts: dict,
    *,
    prompt_path,
    reproduction_mode: bool,
):
    prompt = build_judge_prompt(
        user_turns, prompt_path=prompt_path, reproduction_mode=reproduction_mode
    )
    messages = [{"role": "user", "content": prompt}]
    return [parse_labels(model.complete(messages, **opts).text) for _ in range(n_samples)]


def label_conversation(
    model: Model,
    conversation: Conversation,
    *,
    judge_prompt_path,
    reproduction_mode: bool = False,
    mode: str = MODE_FULL,
    n_samples: int = 5,
    **opts,
) -> dict[int, str | None]:
    """Return {user_turn_index: voted_label} for the conversation's gold turns.

    ``mode=MODE_FULL`` judges all turns in one prompt (N calls total); ``MODE_TURN``
    judges each gold turn with only turns 1..t visible (N calls per gold turn).
    Sampling variety requires a nonzero ``temperature`` in ``opts``.

    ``judge_prompt_path`` names the judge prompt file; ``reproduction_mode`` requires
    it to hash-match the reference prompt (see ``paper_prompts``).
    """
    if mode not in (MODE_FULL, MODE_TURN):
        raise ValueError(f"unknown mode: {mode!r}")

    turns = _user_turns(conversation)
    targets = [g.user_turn_index for g in conversation.gold_turns]
    result: dict[int, str | None] = {}

    load_opts = {"prompt_path": judge_prompt_path, "reproduction_mode": reproduction_mode}
    if mode == MODE_FULL:
        labelings = _sample_labelings(model, turns, n_samples, opts, **load_opts)
        for t in targets:
            result[t] = majority_vote([lab[t] for lab in labelings if t in lab])
    else:  # MODE_TURN
        for t in targets:
            labelings = _sample_labelings(model, turns[:t], n_samples, opts, **load_opts)
            result[t] = majority_vote([lab[t] for lab in labelings if t in lab])
    return result
