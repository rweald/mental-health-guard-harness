# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Scoring: conversation -> continuous P(unsafe) at the decision token.

The classifier decision is a single-token choice. After the guard prompt the model
emits ``Safety`` then ``:``; the NEXT token is the label — ` Safe` vs ` Unsafe`.
We teacher-force the ``Safety:`` prefix and read the logits at that label position.
Score = P(unsafe) via a 2-way softmax over {safe_id, unsafe_id}.

Split so the numeric core (``p_unsafe_from_logits``) and the prompt rendering
(``render_conversation`` / ``build_prompt``) are unit-testable without the 8B
model or a GPU; ``score_conversation`` / ``score_testset`` add the model calls and
are exercised on CPU against a tiny random-weight dummy model.

The guard prompt is not defined here — it is loaded at call time from a
caller-supplied file (see ``paper_prompts``). Every entry point therefore takes a
``guard_prompt_path``; there is no default and no implicit lookup.

MindGuard-8B token ids (defaults): ` Safe`=22291, ` Unsafe`=73067, prefix
``Safety:``=(73037, 25). Parameterized so a small-vocab CPU model works in tests.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .. import paper_prompts
from ..io_utils import open_shared

# Real MindGuard-8B decision tokens; overridable so tiny CPU test models work.
# Qwen3Guard-Gen-8B (MindGuard's own base) shares these exact ids — verified via
# derive_label_tokens — so scoring our fine-tuned adapters needs no id change.
SAFE_ID = 22291
UNSAFE_ID = 73067
PREFIX_IDS = (73037, 25)  # "Safety" then ":"

# Eval lineage: scores land here, never in the training lineage.
DEFAULT_OUTPUT = Path("results/eval/mindguard8b_scores.jsonl")

# Chat roles -> the label used in the rendered conversation block.
_ROLE_LABELS = {"user": "User", "assistant": "Assistant"}


def render_conversation(messages) -> str:
    """Render chat messages into the conversation block that replaces the guard
    prompt's placeholder: one ``Role: content`` block per message (roles capitalized),
    joined by newlines."""
    lines = []
    for msg in messages:
        role = msg["role"]
        label = _ROLE_LABELS.get(role, role.capitalize())
        lines.append(f"{label}: {msg['content']}")
    return "\n".join(lines)


def build_prompt(messages, *, prompt_path, reproduction_mode: bool = False) -> str:
    """Render the guard prompt with the conversation block filled in.

    The prompt is loaded from ``prompt_path`` on every call — the file is tiny and
    reloading it keeps the loader stateless and cache-free."""
    return paper_prompts.load_and_render(
        "guard",
        render_conversation(messages),
        path=prompt_path,
        mode=paper_prompts.mode_for(reproduction_mode),
    )


def p_unsafe_from_logits(logits_row, safe_id=SAFE_ID, unsafe_id=UNSAFE_ID) -> float:
    """2-way softmax P(unsafe) over the {safe, unsafe} logits at the label position.

    Equivalent to softmax over the full-vocab log-softmax restricted to the two
    ids (log-softmax's shared constant cancels), just computed directly.
    """
    pair = torch.stack([logits_row[safe_id], logits_row[unsafe_id]]).float()
    return torch.softmax(pair, dim=-1)[1].item()


@dataclass(frozen=True)
class LabelTokens:
    """The decision tokens for a given tokenizer: teacher-force ``prefix_ids`` (the
    ``Safety:`` stem), then read a 2-way softmax over {safe_id, unsafe_id} at the next
    position. Deriving these from the tokenizer instead of hardcoding them means a
    base-model swap (MindGuard-8B → Qwen3Guard) can't silently desync scoring."""

    safe_id: int
    unsafe_id: int
    prefix_ids: tuple[int, ...]


def derive_label_tokens(tokenizer, stem: str = "Safety:") -> LabelTokens:
    """Locate a tokenizer's decision tokens by encoding the two label targets and
    finding where they first diverge.

    Robust to multi-token label words (e.g. " Unsafe" splitting into pieces): scoring
    reads a single position, so only the FIRST differing token after the shared stem
    matters. The shared prefix becomes ``prefix_ids`` (what scoring teacher-forces);
    the first divergent ids become {safe_id, unsafe_id}. Raises loudly if the two
    targets never diverge (one is a prefix of the other) — that would make a single-
    position 2-way softmax invalid and must not pass silently.
    """
    safe_ids = tokenizer.encode(f"{stem} Safe", add_special_tokens=False)
    unsafe_ids = tokenizer.encode(f"{stem} Unsafe", add_special_tokens=False)
    limit = min(len(safe_ids), len(unsafe_ids))
    k = 0
    while k < limit and safe_ids[k] == unsafe_ids[k]:
        k += 1
    if k == limit:
        raise ValueError(
            f"'{stem} Safe' and '{stem} Unsafe' do not diverge into distinct tokens "
            f"(one tokenizes as a prefix of the other): {safe_ids} vs {unsafe_ids}"
        )
    return LabelTokens(
        safe_id=safe_ids[k], unsafe_id=unsafe_ids[k], prefix_ids=tuple(safe_ids[:k])
    )


def attach_adapter(model, adapter_path):
    """Attach a trained LoRA adapter onto a loaded base model for scoring a fine-tuned
    classifier. Passing the base *model object* (not a path) means
    peft ignores the adapter's recorded base-model name and layers the adapter onto this
    exact instance — so the base loads once and no download is triggered. Kept here, split
    from the eval script's GPU model load, so the adapter path is exercised on CPU."""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path)


@torch.no_grad()
def score_conversation(
    model,
    tokenizer,
    messages,
    *,
    guard_prompt_path,
    reproduction_mode: bool = False,
    safe_id=SAFE_ID,
    unsafe_id=UNSAFE_ID,
    prefix_ids=PREFIX_IDS,
) -> float:
    """Score one conversation -> P(unsafe).

    Applies the chat template to the rendered guard prompt, teacher-forces the
    ``Safety:`` prefix so the next-token logits sit at the label position, and reads
    a 2-way softmax over {safe_id, unsafe_id}.
    """
    content = build_prompt(
        messages, prompt_path=guard_prompt_path, reproduction_mode=reproduction_mode
    )
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    prefix = torch.tensor([list(prefix_ids)], dtype=enc["input_ids"].dtype)
    input_ids = torch.cat([enc["input_ids"], prefix], dim=1).to(model.device)
    attn = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attn).logits[0, -1]
    return p_unsafe_from_logits(logits, safe_id=safe_id, unsafe_id=unsafe_id)


def score_testset(
    model,
    tokenizer,
    dataset,
    output_path=DEFAULT_OUTPUT,
    *,
    guard_prompt_path,
    reproduction_mode: bool = False,
    safe_id=SAFE_ID,
    unsafe_id=UNSAFE_ID,
    prefix_ids=PREFIX_IDS,
    limit=None,
) -> Path:
    """Score every dataset row and write the seam-contract JSONL, one object per line:

    ``{"index", "p_unsafe", "label", "annotator_labels"}``.

    ``label`` / ``annotator_labels`` are carried straight through — this module
    never interprets them (the metrics module owns label interpretation).
    """
    output_path = Path(output_path)
    with open_shared(output_path) as f:
        for index, row in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            p_unsafe = score_conversation(
                model,
                tokenizer,
                row["prompt"],
                guard_prompt_path=guard_prompt_path,
                reproduction_mode=reproduction_mode,
                safe_id=safe_id,
                unsafe_id=unsafe_id,
                prefix_ids=prefix_ids,
            )
            record = {
                "index": index,
                "p_unsafe": p_unsafe,
                "label": row["label"],
                "annotator_labels": row["annotator_labels"],
            }
            f.write(json.dumps(record) + "\n")
    return output_path
