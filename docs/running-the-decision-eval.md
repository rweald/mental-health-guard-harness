# Running the decision-model eval

Decision models (Jev, Clef, OpenAI Decisions, …) return a probability per allowed answer
instead of generating text, so they are scored through OpenRouter's Decisions endpoint
rather than the local logit scorer. This is evaluation only — nothing is trained, and the
decision spec is written once and frozen before the testset is scored.

Every row is sent once with two questions, and each question gets its own scores file in
the usual seam format, so `report_metrics.py` works on either:

| Question        | Type     | Score                 | File                         |
| --------------- | -------- | --------------------- | ---------------------------- |
| `risk_category` | `choice` | `1 − P(safe)`         | `<model>__choice3_scores.jsonl` |
| `unsafe`        | `noul`   | `P(unsafe)`           | `<model>__yesno_scores.jsonl`   |

Both are first-class results; neither is picked over the other after the fact.

**Every scoring command below makes paid OpenRouter calls.** It needs `OPENROUTER_API_KEY`.

## Arms

`scripts/eval/score_decisions.py --model <key>`:

| Key            | OpenRouter id                     | Pinned provider |
| -------------- | --------------------------------- | --------------- |
| `jev-1.14`     | `typesafe/jev-1.14`               | TypeSafe        |
| `clef`         | `cloudflare/clef`                 | Cloudflare      |
| `clef-flash`   | `cloudflare/clef-flash`           | Cloudflare      |
| `gpt-6-luna`   | `openai/gpt-6-luna-decisions`     | OpenAI          |
| `decider-v1.2` | `perplexity/pplx-decider-v1.2`    | Perplexity      |
| `decision-1`   | `microsoft/microsoft-decision-1`  | Azure           |

Every arm denies data collection and disables provider fallback. ZDR is required for all
arms except GPT-6 Luna, whose only endpoint is not ZDR. Clef and Clef-flash are pinned to
Cloudflare's own endpoint because the other hosts serve an fp8 build or a 16k context.

## The decision spec

The question text lives in a user-supplied JSON file, like the guard and judge prompts —
conventionally `prompts/decision_spec.json`, gitignored. `mh_guard_harness.decisions`
fixes its structure (question names, types, option keys) and validates it before any
call. Its hash is recorded in each run's provenance, and a run refuses to resume under a
different spec, so every arm is scored with the same questions.

## 1. Export the testset rows (container, no GPU)

The testset lives in the container's offline HF cache; scoring runs locally.

```bash
./scripts/eval/run.sh python scripts/eval/export_testset_rows.py
```

Writes `results/eval/testset_rows.jsonl`. Row indexes match `score_testset.py`, so the
decision-model scores pair row-for-row with MindGuard-8B and the fine-tuned arms.

## 2. Smoke on synthetic dialogues (paid, cents)

The smoke never touches the testset. It checks mechanics only: response shape, refusals,
context fit, and whether the probabilities are spread out or bunched at 0 and 1. Its
numbers are not results, and the spec is not changed in response to them.

```bash
uv run python scripts/eval/export_smoke_rows.py

for m in jev-1.14 clef clef-flash gpt-6-luna decider-v1.2 decision-1; do
  uv run python scripts/eval/score_decisions.py --model $m \
      --spec prompts/decision_spec.json \
      --rows results/eval/decisions/smoke_rows.jsonl \
      --outdir results/eval/decisions/smoke
done
```

## 3. Score the testset (paid)

```bash
for m in jev-1.14 clef clef-flash gpt-6-luna decider-v1.2 decision-1; do
  uv run python scripts/eval/score_decisions.py --model $m \
      --spec prompts/decision_spec.json
done
```

Outputs land in `results/eval/decisions/`. Per arm:

- `<model>.responses.jsonl` — every raw response, the expensive artifact
- `<model>__choice3_scores.jsonl`, `<model>__yesno_scores.jsonl` — the seam files
- `<model>.failures.jsonl` — rows not scored, per question, as `error`, `refusal` or `invalid`
- a `.provenance.json` per scores file — spec hash, resolved model version, provider,
  cost, latency, failure counts, and how many rows exceeded Jev's long-context threshold

**Rerunning the same command resumes**: answered rows (refusals included) are skipped, and
only failed requests are retried. `--derive-only` rebuilds the scores files from the
response cache with no API calls.

## 4. Metrics (local)

```bash
uv run python scripts/eval/report_metrics.py \
    --scores results/eval/decisions/jev-1.14__choice3_scores.jsonl
```

Rows listed in the failures file are absent from the scores file, so these metrics treat
them as excluded. Check each provenance's `failures` before reading the numbers.
