# Decision-spec tuning (interim)

Interim work on improving the decision spec's harm-to-others results for Jev 1.14 and
Clef-flash. It sits on top of the decision-model harness but is deliberately kept out of
it: nothing here is imported by `mh_guard_harness`, and nothing in the package depends on
it.

Iteration uses synthetic dialogues only, split by conversation into a tuning set and a
holdout set (`export_dev_rows.py`). Spec versions are scored with the harness's own
`scripts/eval/score_decisions.py` into one folder per version, then compared and their
misses read with `report_dev.py`. The holdout is scored once with the final spec.

The testset is not used here. Re-scoring it with a tuned spec is a separate step that
needs explicit approval first.

Outputs go under `results/spec_tuning/`.
