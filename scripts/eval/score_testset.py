# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Score a model over an evaluation set — a published baseline model, or a fine-tuned
adapter layered onto its base. Container / GPU.

Loads the model + tokenizer and the testset from the shared offline HF cache, runs the
single-token teacher-forced scoring over every row, and writes the seam-contract JSONL
(`{index, p_unsafe, label, annotator_labels}`) into the eval lineage under results/eval/.
With --adapter, a trained LoRA adapter is layered onto --model before scoring. Metrics are
computed separately in the local torch-free env (scripts/eval/report_metrics.py) — the
container has torch/transformers/datasets but not sklearn/scipy.

How to run it — the published-model baseline, the smoke, and scoring fine-tuned adapters —
is in docs/running-the-eval.md.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from mh_guard_harness import paper_prompts
from mh_guard_harness.modeling import scoring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="swordhealth/MindGuard-8B")
    ap.add_argument("--adapter", default=None,
                    help="path to a trained LoRA adapter to score on top of --model "
                         "(e.g. results/train/adapters/<judge>__seed<N>)")
    ap.add_argument("--dataset", default="swordhealth/MindGuard-testset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", default=str(scoring.DEFAULT_OUTPUT))
    ap.add_argument("--limit", type=int, default=None, help="score only the first N rows")
    ap.add_argument("--guard-prompt", required=True,
                    help="path to the guard prompt file to score with (for a fine-tuned "
                         "adapter this must be the file it was trained against)")
    ap.add_argument("--reproduction-mode", action="store_true",
                    help="require the guard prompt to hash-match the reference prompt")
    args = ap.parse_args()

    paper_prompts.emit_reproduction_notice(args.reproduction_mode)

    # Load and validate the prompt before the 8B model goes onto the GPU, so a bad path
    # or a failed reproduction check costs a second rather than a model load.
    guard_prompt = paper_prompts.load(
        "guard",
        path=args.guard_prompt,
        mode=paper_prompts.mode_for(args.reproduction_mode),
    )
    print(f"guard prompt: {guard_prompt.filename} "
          f"({guard_prompt.source}, sha256 {guard_prompt.sha256[:12]})", flush=True)

    print(f"loading tokenizer + model: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter:
        print(f"attaching LoRA adapter: {args.adapter}", flush=True)
        model = scoring.attach_adapter(model, args.adapter)
    model.eval()

    # Guard: confirm the real tokenizer still maps to the decision tokens the scorer
    # assumes (defaults were validated for MindGuard-8B). Fail loud if the model swaps.
    safe_ids = tok.encode("Safety: Safe", add_special_tokens=False)
    unsafe_ids = tok.encode("Safety: Unsafe", add_special_tokens=False)
    assert (
        safe_ids[-1] == scoring.SAFE_ID
        and unsafe_ids[-1] == scoring.UNSAFE_ID
        and tuple(safe_ids[:-1]) == scoring.PREFIX_IDS
    ), f"decision-token mismatch: safe={safe_ids} unsafe={unsafe_ids} — update scoring ids"

    ds = load_dataset(args.dataset, split=args.split)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"scoring {n} rows -> {args.output}", flush=True)
    out = scoring.score_testset(
        model, tok, ds,
        output_path=args.output,
        guard_prompt_path=args.guard_prompt,
        reproduction_mode=args.reproduction_mode,
        limit=args.limit,
    )

    # Provenance sidecar: which model, which adapter, and which prompt produced these
    # scores. Without the prompt record a score file is not interpretable later.
    prov = Path(out).with_suffix(".provenance.json")
    prov.write_text(json.dumps({
        "model": args.model,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit,
        "guard_prompt": paper_prompts.provenance(guard_prompt),
    }, indent=2))
    print(f"done. wrote {out} (provenance: {prov})", flush=True)


if __name__ == "__main__":
    main()
