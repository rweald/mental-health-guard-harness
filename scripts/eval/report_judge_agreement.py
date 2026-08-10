# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Step: judge-agreement (local). Compare each judge's turn labels to the clinician
gold and print the full analysis: agreement + directional errors (with bootstrap CIs),
per-category recall, the full-vs-turn context effect, and pairwise significance.

    uv run python scripts/eval/report_judge_agreement.py

Reads results/eval/judge_labels/<judge>__<mode>.jsonl (from run_judge_agreement.py).
"""

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

from mh_guard_harness.agreement import agreement
from mh_guard_harness.analysis import (
    annotator_pairwise_kappa,
    bootstrap_kappa_ci,
    context_effect,
    judge_vs_annotators_kappa,
    mcnemar_compare,
)

HUMAN_FLEISS = 0.568  # inter-annotator ceiling on the eval set (reference line)


def _load_keyed(path):
    """-> {(conv_id, user_turn_index): (judge_label, gold_label)}."""
    d = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        d[(r["conv_id"], r["user_turn_index"])] = (r["judge_label"], r["gold_label"])
    return d


def _parallel(keyed, keys):
    return [keyed[k][0] for k in keys], [keyed[k][1] for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/eval/judge_labels")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples for CIs")
    ap.add_argument("--conversations", default="results/eval/testset_conversations.json",
                    help="source of per-annotator labels for the human-band comparison")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.dir) / "*.jsonl")))
    if not files:
        raise SystemExit(f"no label files in {args.dir} — run run_judge_agreement.py first")

    data = {}  # (judge, mode) -> keyed dict
    for fp in files:
        judge, mode = Path(fp).stem.split("__", 1)
        data[(judge, mode)] = _load_keyed(fp)

    # 1. Agreement table with directional errors + bootstrap CI on 3-way kappa.
    hdr = (f"{'judge / mode':26} {'n':>5} {'cov':>4} {'kappa3 [95% CI]':>20} "
           f"{'kappaB':>7} {'acc':>5} {'rec':>5} {'fnr':>5} {'fpr':>5}")
    print(hdr)
    print("-" * len(hdr))
    for (judge, mode), keyed in sorted(data.items()):
        j, g = _parallel(keyed, list(keyed))
        a = agreement(j, g)
        lo, hi = bootstrap_kappa_ci(j, g, n_boot=args.boot)
        ci = f"{a['kappa_3way']:.3f} [{lo:.2f},{hi:.2f}]"
        print(f"{judge + ' / ' + mode:26} {a['n_scored']:>5} {a['coverage']:>4.2f} "
              f"{ci:>20} {a['kappa_binary']:>7.3f} {a['accuracy']:>5.2f} "
              f"{a['recall']:>5.2f} {a['fnr']:>5.2f} {a['fpr']:>5.2f}")
    print(f"\nreference: inter-annotator Fleiss kappa ~ {HUMAN_FLEISS} (human ceiling)")

    # 1b. Judge as a 4th rater: does it agree with each clinician as much as they agree
    # with each other? Uses the individual annotator labels (not the derived majority),
    # so it avoids the circularity of comparing the judge to a gold built from them.
    ann_map = {}
    all_annotators = []
    conv_path = Path(args.conversations)
    if conv_path.exists():
        for c in json.loads(conv_path.read_text()):
            for gt in c["gold_turns"]:
                ann_map[(c["id"], gt["user_turn_index"])] = gt["annotator_labels"]
                all_annotators.append(gt["annotator_labels"])
    if all_annotators:
        band = annotator_pairwise_kappa(all_annotators)
        pw = "  ".join(f"{k} {v:.2f}" for k, v in band["pairwise"].items())
        print("\nannotator-pool agreement (judge as a 4th rater vs each clinician):")
        print(f"  human pairwise kappa: {pw}   -> band [{band['min']:.2f}, {band['max']:.2f}]")
        print(f"  {'judge / mode':26} {'j_vs_1':>7} {'j_vs_2':>7} {'j_vs_3':>7} "
              f"{'mean':>6} {'in-band':>8}")
        for (judge, mode), keyed in sorted(data.items()):
            keys = [k for k in keyed if k in ann_map]
            jv = judge_vs_annotators_kappa(
                [keyed[k][0] for k in keys], [ann_map[k] for k in keys]
            )
            per, mean = jv["per"], jv["mean"]
            inband = ("n/a" if math.isnan(mean)
                      else "yes" if mean >= band["min"] else "no")
            print(f"  {judge + ' / ' + mode:26} {per['judge_vs_1']:>7.2f} "
                  f"{per['judge_vs_2']:>7.2f} {per['judge_vs_3']:>7.2f} "
                  f"{mean:>6.2f} {inband:>8}")

    # 2. Per-category recall (did the judge catch each kind of risk?).
    print("\nper-category recall (fraction of gold unsafe turns flagged unsafe):")
    print(f"  {'judge / mode':26} {'self_harm':>10} {'harm_to_others':>15}")
    for (judge, mode), keyed in sorted(data.items()):
        j, g = _parallel(keyed, list(keyed))
        rc = agreement(j, g)["recall_by_category"]
        print(f"  {judge + ' / ' + mode:26} {rc['self_harm']:>10.2f} "
              f"{rc['harm_to_others']:>15.2f}")

    # 3. Context effect: full-vs-turn kappa gap per judge (needs both modes).
    by_judge = defaultdict(dict)
    for (judge, mode), keyed in data.items():
        by_judge[judge][mode] = keyed
    ctx = {jm: modes for jm, modes in by_judge.items() if {"full", "turn"} <= set(modes)}
    if ctx:
        print("\ncontext effect (kappa3 full - turn; positive = future context helps):")
        for judge, modes in sorted(ctx.items()):
            keys = sorted(set(modes["full"]) & set(modes["turn"]))
            jf = [modes["full"][k][0] for k in keys]
            jt = [modes["turn"][k][0] for k in keys]
            gold = [modes["full"][k][1] for k in keys]
            ce = context_effect(jf, jt, gold)
            print(f"  {judge:20} full {ce['kappa_full']:.3f}  turn {ce['kappa_turn']:.3f}"
                  f"  delta {ce['delta']:+.3f}")

    # 4. Pairwise significance within each mode (McNemar on per-turn correctness).
    modes = sorted({m for _, m in data})
    for mode in modes:
        judges = sorted(j for (j, m) in data if m == mode)
        pairs = [(a, b) for i, a in enumerate(judges) for b in judges[i + 1:]]
        if not pairs:
            continue
        print(f"\npairwise agreement-with-gold significance ({mode} mode, McNemar):")
        for a, b in pairs:
            ka, kb = data[(a, mode)], data[(b, mode)]
            keys = sorted(set(ka) & set(kb))
            ja = [ka[k][0] for k in keys]
            jb = [kb[k][0] for k in keys]
            gold = [ka[k][1] for k in keys]
            r = mcnemar_compare(ja, jb, gold)
            print(f"  {a} vs {b}: {a} right/{b} wrong={r['a_only']}, "
                  f"{b} right/{a} wrong={r['b_only']}, p={r['pvalue']:.4f}")


if __name__ == "__main__":
    main()
