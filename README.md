# mental-health-guard-harness

A research harness for training and evaluating **turn-level safety classifiers** on
multi-turn mental health conversations.

The approach is distillation. A large judge model reads a full synthetic conversation
between a patient model and a clinician model and labels every patient turn for safety. A
much smaller model is then fine-tuned to make that same call from a single turn in
isolation — the small model being the one cheap enough to actually run on every turn of
every conversation.

This repository holds the machinery for that loop end to end:

- **Generation** — a two-agent patient/clinician conversation loop driven by a scenario library
- **Labeling** — a full-context judge with N-sample majority voting per turn
- **Training** — LoRA fine-tuning, locally or fanned out across cloud GPUs
- **Evaluation** — teacher-forced single-token scoring, AUROC, FPR at fixed TPR, and
  per-subcategory breakdowns

Model access goes through a ~30-line adapter layer rather than an orchestration framework,
so which model fills which role is a config change.

## Relationship to the original work

The method reproduced here is **MindGuard** (Farinhas et al.,
[arXiv:2602.00950](https://arxiv.org/abs/2602.00950)). That paper, its released models, and
its clinician-labeled evaluation set are licensed CC BY-NC-SA 4.0. This is an independent
implementation, not a fork: no code, weights, evaluation data, or prompt text from that work
is included here.

**This repository does not include the reference prompts or the evaluation data, and is not
sufficient to reproduce the paper on its own.** Both are obtainable from the original
authors' released material; what lives here is the harness that runs against them. See
[`prompts/README.md`](prompts/README.md).

## Setup

Requires Python 3.12 (the interpreter is managed by asdf via `.tool-versions`; uv never
downloads its own).

```bash
uv sync
uv run pytest
```

The harness ships **no prompt text**. Supply two files and pass their paths explicitly —
there is no default and no implicit lookup relative to the working directory:

```bash
# score a model over an eval set
uv run python scripts/eval/score_testset.py \
    --model <model-id> \
    --guard-prompt prompts/guard.txt

# label dialogues with a judge
uv run python scripts/generate/label_dialogues.py \
    --dialogues results/generated/dialogues.jsonl \
    --judge-prompt prompts/judge.txt

# fine-tune one adapter
uv run python scripts/train/run_finetune.py \
    --judge <name> --labels <labels.jsonl> --seed 0 \
    --guard-prompt prompts/guard.txt
```

Add `--reproduction-mode` to require a prompt to hash-match the published reference version;
the default accepts any prompt carrying the right placeholder and records its hash in the
run's provenance. `prompts/` is a documented convention, not a search path.

## Adapting it

[`docs/adapting-this-harness.md`](docs/adapting-this-harness.md) documents the seams — the
`Model` protocol, the prompt contract, the scenario schema, and the JSONL seam between
scoring and metrics — along with which parts are load-bearing for the method and which are
free parameters.

Operator runbooks: [running the eval](docs/running-the-eval.md) ·
[running the fine-tune](docs/running-the-finetune.md).

## Validation lineage

Scoring the published MindGuard-8B through this harness reproduces AUROC **0.9839** against
the paper's reported ~0.982. That provides an end-to-end regression anchor for the scoring
and evaluation implementation — not a proof that each component is individually correct.

The evaluation set is small and heavily imbalanced (on the order of 42 positive examples),
so confidence intervals are wide and headline AUROC saturates. Report spread across seeds.

## Scope and limitations

**Research and evaluation infrastructure, not a clinically validated decision system. Must
not be used as the sole basis for emergency or treatment decisions.**

## License

Copyright 2026 Ryan Weald

Except where otherwise noted, the source code and documentation in this repository are
licensed under Apache-2.0. See [`LICENSE`](LICENSE).

Prompt files you supply are **not** covered by that license. Prompts transcribed from the
original paper carry its CC BY-NC-SA 4.0 terms, and so does work derived from them —
including judge labels and models trained on those labels. The harness prints a reminder
when reproduction mode is used.
