# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Derive and verify Qwen3Guard's safety-decision token ids — a fine-tune prerequisite.

The classifier's score is a 2-way softmax over {Safe, Unsafe} at one label position.
MindGuard-8B's ids are hardcoded in ``modeling.scoring``; Qwen3Guard has its own
tokenizer, so those ids must be re-derived before any training/eval against it — a
wrong id silently invalidates every score. This loads the base tokenizer (no GPU, no
model weights) and prints the derived ids plus their decoded pieces for a human check.

    ./scripts/eval/run.sh python scripts/train/derive_label_ids.py

Runs offline from the shared weight cache. Exits non-zero if the two targets don't
tokenize to a clean divergence.
"""

import argparse

from transformers import AutoTokenizer

from mh_guard_harness.modeling.scoring import derive_label_tokens
from mh_guard_harness.training.config import TrainConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TrainConfig().base_model)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    lt = derive_label_tokens(tok)

    print(f"base model: {args.model}")
    print(f"  full 'Safety: Safe'   -> {tok.encode('Safety: Safe', add_special_tokens=False)}")
    print(f"  full 'Safety: Unsafe' -> {tok.encode('Safety: Unsafe', add_special_tokens=False)}")
    print("derived decision tokens:")
    print(f"  prefix_ids = {lt.prefix_ids}  decoded={tok.decode(lt.prefix_ids)!r}")
    print(f"  safe_id    = {lt.safe_id}  decoded={tok.decode([lt.safe_id])!r}")
    print(f"  unsafe_id  = {lt.unsafe_id}  decoded={tok.decode([lt.unsafe_id])!r}")
    print("\nwire these into scoring when evaluating fine-tuned Qwen3Guard adapters.")


if __name__ == "__main__":
    main()
