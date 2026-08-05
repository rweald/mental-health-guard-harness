# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Compare the fine-tuned arms across judges and seeds after scoring. Local, torch-free.

``report_metrics.py`` reports ONE scores file. This reads all of them and answers the
question the study is actually asking: does the judge that labeled the training data
change the classifier it produces?

Three things per arm, aggregated over seeds:

- **AUROC mean ± sd across seeds** — the spread here is *training* variance (does the
  fine-tune reproduce), which is NOT the uncertainty that decides whether an arm is
  better. That one comes from the bootstrap below, and is ~10x wider.
- **FPR@90/95%TPR and per-subgroup AUROC** — via the same helpers ``report_metrics.py``
  uses, so per-cell numbers match what that script prints.
- **Paired bootstrap between arms**, on seed-averaged scores (averaging collapses
  training noise so the interval reflects testset sampling variance). Run overall and
  per subgroup: the overall test mixes a gain in one category with a loss in another and
  can miss both, so the subgroup split is where a localized effect shows up.

    uv run python scripts/eval/compare_arms.py
    uv run python scripts/eval/compare_arms.py --json > results/eval/step5_comparison.json
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from mh_guard_harness import metrics
from mh_guard_harness.analysis import bootstrap_auroc_diff

JUDGES = ["glm-4.6", "glm-5.2", "deepseek-v4-flash"]
SEEDS = (0, 1, 2)
SUBGROUPS = ("self_harm", "harm_to_others")
DEFAULT_SCORES_DIR = "results/eval"
BASELINE = "results/eval/mindguard8b_scores.jsonl"


def arm_path(scores_dir, judge, seed):
    return Path(scores_dir) / f"{judge}__seed{seed}_scores.jsonl"


def load_arms(scores_dir):
    """{judge: [frame per seed]}, skipping arms whose files aren't all present."""
    arms = {}
    for judge in JUDGES:
        paths = [arm_path(scores_dir, judge, s) for s in SEEDS]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"! {judge}: skipping, missing {len(missing)} of {len(SEEDS)} seeds "
                  f"({missing[0].name}...)")
            continue
        arms[judge] = [metrics.load_scores(p) for p in paths]
    return arms


def per_arm_summary(frames):
    """Seed-wise metrics for one arm, plus mean/sd across seeds."""
    aurocs, fpr90, fpr95 = [], [], []
    subs = {s: [] for s in SUBGROUPS}
    for df in frames:
        aurocs.append(metrics.auroc(df["y_true"], df["p_unsafe"]))
        both = metrics.fpr_at_90_95(df["y_true"], df["p_unsafe"])
        fpr90.append(both["fpr_at_90"]["fpr"])
        fpr95.append(both["fpr_at_95"]["fpr"])
        for s in SUBGROUPS:
            subs[s].append(metrics.subgroup_auroc(df["risk_category"], df["p_unsafe"], s))

    def ms(values):
        a = np.asarray(values, dtype=float)
        # ddof=1: three seeds are a sample of the training-noise distribution, not it.
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
                "seeds": [float(x) for x in a]}

    return {
        "auroc": ms(aurocs),
        "fpr_at_90": ms(fpr90),
        "fpr_at_95": ms(fpr95),
        "subgroup_auroc": {s: ms(v) for s, v in subs.items()},
    }


def seed_averaged(frames):
    """Mean p_unsafe over seeds. Rows are identical across seeds (same testset, same
    order), so a positional mean is well defined; assert rather than trust it."""
    first = frames[0]
    for df in frames[1:]:
        if not first["y_true"].equals(df["y_true"]):
            raise ValueError("seed score files disagree on row order or gold labels")
    return first, np.mean([df["p_unsafe"].to_numpy() for df in frames], axis=0)


def compare(arms, n_boot, seed):
    """Paired bootstrap for every judge pair — overall and within each subgroup."""
    avg = {j: seed_averaged(f) for j, f in arms.items()}
    out = {}
    for a, b in combinations(sorted(arms), 2):
        (df, p_a), (_, p_b) = avg[a], avg[b]
        y = df["y_true"].to_numpy()
        cats = df["risk_category"].to_numpy()
        entry = {"overall": bootstrap_auroc_diff(y, p_a, p_b, n_boot=n_boot, seed=seed)}
        for sub in SUBGROUPS:
            keep = (cats == sub) | (cats == "safe")
            entry[sub] = bootstrap_auroc_diff(
                (cats[keep] == sub).astype(int), p_a[keep], p_b[keep],
                n_boot=n_boot, seed=seed,
            )
        out[f"{a} - {b}"] = entry
    return out


def saturation(frames):
    """Share of scores pinned at 0/1 vs graded. A model that saturates gives tied scores
    and blunts FPR@high-TPR, which is exactly the metric this study leans on."""
    pinned, graded = [], []
    for df in frames:
        p = df["p_unsafe"].to_numpy()
        pinned.append(float(((p <= 1e-6) | (p >= 1 - 1e-6)).mean()))
        graded.append(float(((p > 0.01) & (p < 0.99)).mean()))
    return {"pinned_at_0_or_1": float(np.mean(pinned)), "graded_0.01_0.99": float(np.mean(graded))}


def build_report(scores_dir, n_boot, seed):
    arms = load_arms(scores_dir)
    if len(arms) < 1:
        raise SystemExit(f"no complete arms found under {scores_dir} — run the eval sweep first")
    report = {
        "arms": {j: per_arm_summary(f) for j, f in arms.items()},
        "saturation": {j: saturation(f) for j, f in arms.items()},
        "bootstrap": compare(arms, n_boot, seed) if len(arms) > 1 else {},
        "n_boot": n_boot,
    }
    baseline = Path(BASELINE)
    if baseline.exists():
        df = metrics.load_scores(baseline)
        # Reference point only: the published model trained on a different dialogue set,
        # so this is not a like-for-like comparison with the arms above.
        report["published_baseline_auroc"] = metrics.auroc(df["y_true"], df["p_unsafe"])
    return report


def print_report(report):
    print("=== AUROC by judge (mean ± sd over seeds; spread = TRAINING variance) ===")
    for judge, m in report["arms"].items():
        a = m["auroc"]
        print(f"  {judge:20} {a['mean']:.5f} ± {a['sd']:.5f}   "
              f"seeds: {', '.join(f'{x:.5f}' for x in a['seeds'])}")
    if "published_baseline_auroc" in report:
        print(f"  {'MindGuard-8B (pub.)':20} {report['published_baseline_auroc']:.5f}   "
              "[different training data — not like-for-like]")

    print("\n=== FPR at TPR targets (lower is better) ===")
    for key, name in (("fpr_at_90", "FPR@90"), ("fpr_at_95", "FPR@95")):
        print(f"  {name}:")
        for judge, m in report["arms"].items():
            print(f"    {judge:20} {m[key]['mean']:.4f} ± {m[key]['sd']:.4f}")

    print("\n=== Subgroup AUROC ===")
    for sub in SUBGROUPS:
        print(f"  {sub}:")
        for judge, m in report["arms"].items():
            s = m["subgroup_auroc"][sub]
            print(f"    {judge:20} {s['mean']:.5f} ± {s['sd']:.5f}")

    if report["bootstrap"]:
        print(f"\n=== Paired bootstrap over testset rows "
              f"({report['n_boot']} resamples, seed-averaged scores) ===")
        print("    (this is the SAMPLING uncertainty — the one that decides 'is A better than B')")
        for pair, entry in report["bootstrap"].items():
            print(f"  {pair}")
            for scope in ("overall", *SUBGROUPS):
                r = entry[scope]
                star = " *" if not (r["ci_low"] <= 0 <= r["ci_high"]) else ""
                print(f"    {scope:16} Δ={r['delta']:+.5f}  "
                      f"95% CI [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]  "
                      f"p={r['pvalue']:.4f}{star}")
        print("    * CI excludes zero")

    print("\n=== Score saturation (mean over seeds) ===")
    for judge, s in report["saturation"].items():
        print(f"  {judge:20} pinned@0/1={s['pinned_at_0_or_1']:6.2%}  "
              f"graded={s['graded_0.01_0.99']:6.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default=DEFAULT_SCORES_DIR)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed (reproducibility)")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON instead")
    args = ap.parse_args()

    report = build_report(args.scores_dir, args.n_boot, args.seed)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
