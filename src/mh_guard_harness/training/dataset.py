# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Turn judge-labeled dialogues into supervised fine-tuning examples.

Each labeled USER turn becomes one training example: the past-only conversation
prefix ending AT that turn, rendered through the SAME guard prompt the eval scorer
uses (``modeling.scoring.build_prompt``), paired with the two-line target the
classifier emits.

The guard prompt is loaded from a caller-supplied file by ``build_prompt``; this
module just forwards the path it is given. Training and scoring MUST be handed the
same prompt file — a student trained against one prompt and scored against another
is measuring the mismatch, not the model.

The asymmetry here IS the method (paper §5): the judge produced its label with
FULL-conversation context (later turns visible), but the student only ever SEES the
past. So the model learns to predict a full-context label from past context alone —
matching how it is deployed and how ``score_conversation`` scores it. The builder
enforces past-only context structurally: a prefix never includes a turn beyond the
one being labeled.

Pure text — no torch, tokenizer, or token ids. Tokenization and assistant-only loss
masking live in ``train.py``; the Safe/Unsafe token ids matter only at scoring time.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..generation import Dialogue
from ..modeling.scoring import build_prompt

# Judge label -> (Safety line, Categories line) for the §7.4 two-line target.
# safe -> None; the two unsafe axes map to the paper's S1 (self-harm) / S2 (threats).
_LABEL_TO_TARGET = {
    "safe": ("Safe", "None"),
    "unsafe_self_harm_risk": ("Unsafe", "S1"),
    "unsafe_threats_to_others": ("Unsafe", "S2"),
}


def render_target(judge_label: str) -> str:
    """The two-line completion the classifier is trained to emit:
    line 1 ``Safety: Safe|Unsafe``, line 2 ``Categories: S1|S2|None``."""
    try:
        safety, category = _LABEL_TO_TARGET[judge_label]
    except KeyError:
        raise ValueError(f"unknown judge label: {judge_label!r}")
    return f"Safety: {safety}\nCategories: {category}"


@dataclass
class TrainExample:
    """One supervised example: a guard prompt over past-only context and its target."""

    dialogue_id: str
    user_turn_index: int
    label: str        # raw judge label, kept for stratification / debugging
    prompt: str       # rendered guard prompt, context ends at this user turn
    completion: str   # the two-line target

    def to_chat(self) -> list[dict]:
        """Chat-format view for a template-based, assistant-only SFT collator."""
        return [
            {"role": "user", "content": self.prompt},
            {"role": "assistant", "content": self.completion},
        ]


def _prefixes(messages: list[dict]) -> dict[int, list[dict]]:
    """Map each 1-based user-turn index -> the message prefix ending AT that user turn
    (past-only: no later turns, and the prefix's final message is that user turn)."""
    out: dict[int, list[dict]] = {}
    seen_users = 0
    for i, m in enumerate(messages):
        if m["role"] == "user":
            seen_users += 1
            out[seen_users] = messages[: i + 1]
    return out


def build_examples(
    dialogues: list[Dialogue],
    label_rows: list[dict],
    *,
    guard_prompt_path,
    reproduction_mode: bool = False,
) -> list[TrainExample]:
    """Join ``dialogues`` to one judge's ``label_rows`` (each ``{dialogue_id,
    user_turn_index, judge_label}``) into one example per labeled user turn.

    Only turns present in BOTH the dialogues and the labels are emitted, in a fixed
    order — dialogue-file order, then ascending turn index — so every label set and
    seed sees the identical example sequence before any shuffle. That determinism is
    what keeps the judge the only variable across the three fine-tunes.
    """
    labels: dict[tuple[str, int], str] = {}
    for r in label_rows:
        label_key = (r["dialogue_id"], r["user_turn_index"])
        if label_key in labels:
            # Silently keeping one of them would train on an arbitrary resolution and
            # break the identical-example-sequence guarantee across label sets.
            raise ValueError(f"duplicate label row for {label_key}")
        labels[label_key] = r["judge_label"]
    examples: list[TrainExample] = []
    for dialogue in dialogues:
        key = dialogue.key()
        for turn_index, prefix in sorted(_prefixes(dialogue.messages).items()):
            label = labels.get((key, turn_index))
            if label is None:
                continue
            examples.append(
                TrainExample(
                    dialogue_id=key,
                    user_turn_index=turn_index,
                    label=label,
                    prompt=build_prompt(
                        prefix,
                        prompt_path=guard_prompt_path,
                        reproduction_mode=reproduction_mode,
                    ),
                    completion=render_target(label),
                )
            )
    return examples
