# Adapting this harness

This harness was built to run one specific study, but the parts that made that study work
are the parts you are most likely to want to change: which models fill which role, what the
prompts say, what conversations get generated, and what you score against. This document
covers those seams — what each one guarantees, and what breaks if you cross it.

## Validation anchor

Scoring the published MindGuard-8B through this harness reproduces AUROC **0.9839** against
the paper's reported ~0.982. That provides an end-to-end regression anchor for the scoring
and evaluation implementation: prompt rendering, teacher-forced single-token scoring,
threshold sweep and metric computation together produce the expected number on a known
model and a known dataset.

It is an anchor, not a proof that each component is individually correct. A compensating
pair of errors would still land near the right number. Treat it as the check that catches
gross breakage — which is what it is good at — and rely on the unit tests for the parts.

## Swapping models

Every pipeline step takes a `Model`, which is a one-method protocol:

```python
class Model(Protocol):
    def complete(self, messages: list[Message], **opts: Any) -> Completion: ...
```

`Completion` carries `.text` plus `.raw`, the untouched provider response. An "arm" config
names which adapter fills each role — patient, clinician, judge — so swapping a model is a
config change rather than a code change.

`OpenRouterModel` wraps the OpenAI SDK against an OpenAI-compatible base URL, which covers
a large share of hosted models via a base-URL and model-id swap. "OpenAI-compatible" is a
loose label in practice, so the real requirement is narrower: the endpoint must implement
the fields this adapter actually reads. If it accepts a chat-style `messages` list and
returns a text completion, it will work; per-model knobs (temperature, reasoning effort)
ride along in `default_opts` and per-call `opts`.

One deliberate constraint: `complete` returns **one** sample per call. Callers that need N
samples loop. A provider that quietly ignores an `n` parameter would otherwise collapse a
majority vote to a single sample without anyone noticing.

**Log-probability support is required of the locally scored model, not of the API
adapters.** Scoring reads logits at a single label position from a local `transformers`
model object. The API adapters never need log-probs — they only generate text.

## Bringing your own prompts

The harness ships no prompt text; both prompts are loaded from files you supply. See
[`prompts/README.md`](../prompts/README.md) for the contract. Two rules matter downstream:

- **Change the guard prompt → retrain.** The classifier is trained *and* scored through this
  prompt. A model trained against one and scored against another measures the mismatch.
- **Change the judge prompt → relabel.** Labels are the judge prompt's output; existing
  label sets do not carry over.

Each run records the SHA-256 of the prompt it used — in `run.json` for training, in a
`.provenance.json` sidecar for scoring and labeling. When results look strange, comparing
those hashes is the fastest way to rule out a prompt mismatch.

## Bringing your own scenario library

Scenarios are the generation unit. `scenarios.errors()` returns a list of structural
problems (empty means valid) and is the authority on what a scenario must satisfy:

- `risk_category` from the known set, with a `subcategory` valid *for that category*
- a `trajectory` from the known set
- a non-empty `turn_script` whose entries are all valid risk tags
- a `safe` scenario may include BORDERLINE turns but **never** UNSAFE
- a non-safe scenario must actually reach BORDERLINE or UNSAFE somewhere

That safe-scenario rule is deliberately more permissive than the method it came from, which
keeps safe scenarios entirely SAFE. Validating leniently means an authoring model that
incidentally scripts a BORDERLINE turn in a safe scenario produces output worth keeping
rather than output worth paying to regenerate — and ambiguous-but-safe dialogues are exactly
the hard negatives that FPR-at-high-TPR rewards. The authoring prompt still asks for the
stricter ideal; only the validator is lenient.

Note what the turn tags are and are not: they steer *generation*. They are not training
labels. The judge labels every turn independently, and its labels are what training sees.

### Changing the vocabularies

`RISK_CATEGORIES`, `SUBCATEGORIES`, `TRAJECTORIES` and `RISK_TAGS` in `scenarios.py` are
meant to be edited — they encode one population and one taxonomy. But unlike the persona
attribute list, they are only *partly* derived: each has companion sites in `authoring.py`
and `generation.py` that must change with it, and **every failure mode is silent**. Add a
subcategory without its authoring weight and it validates fine and is never authored; add
a risk category without a `_tag_guidance` branch and its UNSAFE turns are generated as
ordinary safe talk.

The module docstring in `scenarios.py` lists the companion site for each vocabulary. Read
it before editing one — it is the difference between a change that works and a change that
quietly produces a library missing the thing you added.

Two related traps: `errors()` is the only validator, so a semantic rule for a new category
has to be written there by hand; and scenario risk categories are a *different vocabulary*
from the judge's labels, which the judge prompt emits and
`training/dataset.py` maps to training targets.

### Persona fields

The `Persona` dataclass in `scenarios.py` *is* the persona schema — there is no separate
schema file and no configuration for it. Edit the field list directly; that is the
supported path. The attributes it ships with suit general adult mental-health support, and
a different population will want different ones.

Everything downstream is derived from the dataclass — the attribute list handed to the
authoring model, the persona block rendered into the patient prompt, and JSON in both
directions — and nothing in the codebase reaches for a persona field by name, so adding or
removing one needs no other edit. Existing scenario libraries still load afterwards:
unrecognised keys are dropped and absent ones take their defaults. The `Persona` docstring
covers the few constraints (every field needs a default; narrative fields are listed
separately from attributes) and the comparability caveat.

## Bringing your own eval set

The scorer and the metrics code meet at a JSONL file, one object per line. That seam is the
contract; anything producing it can be scored.

| Field              | Type      | Meaning                                          |
| ------------------ | --------- | ------------------------------------------------ |
| `schema_version`   | int       | Version of this contract                         |
| `index`            | —         | Stable unique row id                             |
| `p_unsafe`         | float     | Finite, in `[0, 1]`                              |
| `label`            | enum      | Gold label; document your allowed values         |
| `annotator_labels` | list/dict | Per-annotator labels; document allowed values    |

The scoring module carries `label` and `annotator_labels` straight through without
interpreting them — label interpretation lives entirely in the metrics module. So a new eval
set needs its label vocabulary taught to the metrics code, not the scorer.

## Method fidelity vs free parameters

Some things are load-bearing for the method; others are choices this particular study made
and you can change freely.

**Load-bearing.** The judge labels with *full-conversation* context while the student sees
only the past. That asymmetry is the method, not an oversight — the student learns to predict
a full-context judgement from past context alone, which is how it is deployed. The
distillation direction (large judge teaches small turn-level classifier) is likewise the
point. And training and scoring must share a guard prompt.

**Free.** Base model and adapter shape, judge model, patient and clinician models, scenario
library size and mix, sample count for the majority vote, seeds. The frozen `TrainConfig`
exists so a *comparison* holds these fixed — not because the values are sacred. Change them
deliberately and record what you changed; `TrainConfig.fingerprint()` is there to make
"were these runs actually comparable?" answerable after the fact.

## Limits worth stating up front

The clinician-labeled evaluation set is small and heavily imbalanced — on the order of 42
positive examples. Confidence intervals on anything conditioned on positives are wide, and
headline AUROC saturates near the top of its range, so differences between good models
compress into a narrow band.

The practical consequences: report spread across seeds rather than a single run, expect
plausible-looking differences to be within noise, and prefer FPR at high TPR and
per-subcategory breakdowns over headline AUROC when comparing models that are all already
good. A difference that survives across seeds is worth more than a larger difference that
does not.

This is research and evaluation infrastructure. It is not a clinically validated decision
system and must not be used as the sole basis for emergency or treatment decisions.
