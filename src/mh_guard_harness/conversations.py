# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct the MindGuard testset's per-turn rows into full conversations, and
align each gold-labeled turn to its position — the input unit for §7.2 judging.

The testset stores 1134 rows, each a conversation prefix ending on the target user
turn plus that turn's clinician label. Rows nest: a length-5 prefix row is the same
conversation as a length-7 row that extends it. Grouping by *maximal* prefix yields
67 conversations; every gold turn maps to a (conversation, user-turn index). A few
prefixes (generic openers) are shared by branching conversations — the same clinician
label then attaches to that turn in each branch, which is correct: under full-context
judging the branches legitimately differ on that turn because their futures differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _sig(prompt: list[dict]) -> tuple[tuple[str, str], ...]:
    """Hashable (role, content) sequence — a conversation-prefix key."""
    return tuple((m["role"], m["content"]) for m in prompt)


@dataclass
class GoldTurn:
    user_turn_index: int  # 1-based position among USER turns (USER_TURN_N in §7.2)
    msg_index: int  # 0-based index into Conversation.messages
    label: str
    annotator_labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Conversation:
    id: str
    messages: list[dict]  # full [{role, content}, ...], ends on the deepest turn
    gold_turns: list[GoldTurn]  # only turns that carry a clinician label

    @property
    def num_user_turns(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")


def from_dicts(payload) -> list[Conversation]:
    """Rebuild Conversation objects from ``asdict``-serialized JSON (the extractor's
    output), so the local judge step can load conversations without the dataset."""
    return [
        Conversation(
            id=c["id"],
            messages=c["messages"],
            gold_turns=[GoldTurn(**g) for g in c["gold_turns"]],
        )
        for c in payload
    ]


def reconstruct(rows) -> list[Conversation]:
    """Group per-turn rows into maximal conversations with aligned gold turns.

    ``rows`` is any iterable of dicts with ``prompt`` (list of {role, content}),
    ``label``, and optional ``annotator_labels``.
    """
    gold_by_sig: dict[tuple, dict] = {}
    for r in rows:
        gold_by_sig[_sig(r["prompt"])] = {
            "label": r["label"],
            "annotator_labels": r.get("annotator_labels", {}),
            "prompt": r["prompt"],
        }

    all_sigs = set(gold_by_sig)
    # Maximal = not a strict prefix of any other row's prompt.
    maximal = [
        s for s in all_sigs if not any(o != s and o[: len(s)] == s for o in all_sigs)
    ]

    conversations: list[Conversation] = []
    for i, m in enumerate(sorted(maximal)):
        messages = [{"role": role, "content": content} for role, content in m]
        gold_turns: list[GoldTurn] = []
        for end in range(1, len(m) + 1):
            role, _ = m[end - 1]
            if role != "user":
                continue
            prefix = m[:end]
            g = gold_by_sig.get(prefix)
            if g is None:
                continue
            gold_turns.append(
                GoldTurn(
                    user_turn_index=sum(1 for rr, _ in prefix if rr == "user"),
                    msg_index=end - 1,
                    label=g["label"],
                    annotator_labels=g["annotator_labels"],
                )
            )
        conversations.append(
            Conversation(id=f"conv_{i:03d}", messages=messages, gold_turns=gold_turns)
        )
    return conversations
