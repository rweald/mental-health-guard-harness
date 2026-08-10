# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune one LoRA adapter: one judge's labels × one seed. Container/GPU.

    ./scripts/eval/run.sh python scripts/train/run_finetune.py \
        --judge glm-5.2 \
        --labels results/generated/labels/glm-5.2__full.jsonl \
        --seed 0

Loads the 8B base onto the GPU — run only with GPU sign-off. Writes the adapter,
run.json (provenance + config fingerprint), and train_log.jsonl under
results/train/adapters/<judge>__seed<N>/.
"""

import argparse

from mh_guard_harness import paper_prompts
from mh_guard_harness.training.config import RunSpec
from mh_guard_harness.training.train import (
    DEFAULT_DIALOGUES,
    adapter_dir,
    run_finetune,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, help="label-set name, e.g. glm-5.2")
    ap.add_argument("--labels", required=True, help="path to a <judge>__<mode>.jsonl label file")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None,
                    help="adapter output dir (default results/train/adapters/<judge>__seed<N>)")
    ap.add_argument("--dialogues", default=DEFAULT_DIALOGUES)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap optimizer steps (smoke/debug); omit for a full run")
    ap.add_argument("--guard-prompt", required=True,
                    help="path to the guard prompt file used to render training examples "
                         "(must be the same file the eval scorer is given)")
    ap.add_argument("--reproduction-mode", action="store_true",
                    help="require the guard prompt to hash-match the reference prompt")
    args = ap.parse_args()

    paper_prompts.emit_reproduction_notice(args.reproduction_mode)

    out = args.out or adapter_dir(args.judge, args.seed)
    run = RunSpec(
        judge=args.judge,
        seed=args.seed,
        label_path=args.labels,
        out_dir=out,
        guard_prompt_path=args.guard_prompt,
        reproduction_mode=args.reproduction_mode,
    )
    steps = f", max_steps={args.max_steps}" if args.max_steps is not None else ""
    print(f"fine-tuning {args.judge} seed {args.seed} -> {out} "
          f"(config {run.config.fingerprint()}{steps})")
    path = run_finetune(run, dialogues_path=args.dialogues, max_steps=args.max_steps)
    print(f"saved adapter to {path}")


if __name__ == "__main__":
    main()
