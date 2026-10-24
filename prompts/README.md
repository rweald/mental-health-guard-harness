# Prompts

This harness does not ship prompt text. Two prompt files are expected here, and neither
is committed — everything in this directory except this README is gitignored.

| File        | What it is                        | Placeholder      |
| ----------- | --------------------------------- | ---------------- |
| `guard.txt` | Turn-level classifier prompt      | `{conversation}` |
| `judge.txt` | Full-context judge labeling prompt | `{transcript}`   |

Each file must contain its placeholder **exactly once**. That is the only structural
requirement — the rest of the prompt is yours to write.

Decision-model scoring takes a third, optional file, `decision_spec.json`: the typed
questions sent to decision models instead of a prompt template. It has no placeholder;
its structure is validated by `mh_guard_harness.decisions` — see
[running the decision-model eval](../docs/running-the-decision-eval.md).

## Why they are not included

The reference versions of both prompts come from MindGuard (Farinhas et al.,
[arXiv:2602.00950](https://arxiv.org/abs/2602.00950)), which is licensed CC BY-NC-SA 4.0.
This harness is Apache-2.0, and the two licences are not compatible, so the prompt text
cannot live in this repository. The harness records only a SHA-256 hash of each reference
prompt — enough to confirm a transcription is correct, not enough to reconstruct one.

## Getting the reference prompts

To reproduce the published method, transcribe them from the [paper](https://arxiv.org/pdf/2602.00950):

- `guard.txt` — Figure 19 (§7.4)
- `judge.txt` — Figure 8 (§7.2)

Then run with `--reproduction-mode`, which fails unless each file hash-matches the
recorded reference value. Whitespace-only differences are forgiven: files are normalized
(NFC, LF endings, no trailing whitespace) before hashing.

Note that prompts transcribed from the paper carry the paper's CC BY-NC-SA 4.0 licence,
and so does work derived from them — including labels produced by the judge prompt and
models trained on those labels. The harness prints a reminder when reproduction mode is
used.

## Using your own prompts

The default (`custom`) mode accepts any prompt with the right placeholder. Its hash is
recorded in each run's provenance, so results stay traceable to the prompt that produced
them. Nothing you write here inherits the paper's licence.

Changing `guard.txt` means retraining — the classifier is trained and scored through this
prompt, and a model trained against one prompt but scored against another measures the
mismatch. Changing `judge.txt` means relabeling.

## Paths are explicit

Nothing is auto-discovered from this directory; it is a convention, not a lookup path.
Every entry point takes the path explicitly (`--guard-prompt`, `--judge-prompt`), so a
run cannot pick up a different prompt because it started in a different directory.
