# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Compare decision-spec versions on the synthetic tuning set. Local, no API calls.

Each ``--run`` is the output directory of score_decisions.py for one spec version over
``tune_rows.jsonl``. Prints, per spec, model and question: AUROC overall and per risk
category, FPR at 95% TPR, and how the three-way answer sorted the harm-to-others turns.
Labels are the GLM-5.2 judge's.

With ``--misses``, also writes a Markdown file per run listing the harm-to-others turns
scored lowest and the safe turns scored highest, with their text, for reading what the
spec gets wrong. Synthetic dialogues only — never point this at testset rows.

    uv run python scripts/spec_tuning/report_dev.py \\
        --run results/spec_tuning/runs/v1 --run results/spec_tuning/runs/v2 --misses 15
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from mh_guard_harness import metrics

MODELS = ("jev-1.14", "clef-flash")
QUESTIONS = ("choice3", "yesno")
DEFAULT_ROWS = "results/spec_tuning/tune_rows.jsonl"


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _spec_sha(run, model):
    prov = json.loads((Path(run) / f"{model}__yesno_scores.provenance.json").read_text())
    return prov["decision_spec"]["sha256"][:12], prov["run"]


def summarize(run, model, question):
    path = Path(run) / f"{model}__{question}_scores.jsonl"
    rep = metrics.evaluate(path)
    out = {
        "n": rep["n"],
        "auroc": rep["auroc"],
        "sh": rep["subgroup_auroc"][metrics.SELF_HARM],
        "hto": rep["subgroup_auroc"][metrics.HARM_TO_OTHERS],
        "fpr95": rep["fpr_at_95"]["fpr"],
    }
    if question == "choice3":
        picks = Counter(r["choice"] for r in _read_jsonl(path)
                        if metrics.risk_category(r["label"]) == metrics.HARM_TO_OTHERS)
        out["hto_choices"] = dict(picks)
    return out


def write_misses(run, model, rows_by_index, k):
    """Lowest-scored harm-to-others positives and highest-scored safe turns, yes/no
    question, with the message assessed and the turn before it."""
    scores = _read_jsonl(Path(run) / f"{model}__yesno_scores.jsonl")
    hto = sorted((r for r in scores
                  if metrics.risk_category(r["label"]) == metrics.HARM_TO_OTHERS),
                 key=lambda r: r["p_unsafe"])[:k]
    safe = sorted((r for r in scores if r["label"] == metrics.SAFE),
                  key=lambda r: -r["p_unsafe"])[:k]
    lines = [f"# Misses: {model} on {Path(run).name}", ""]
    for title, items in (("Harm-to-others turns scored lowest", hto),
                         ("Safe turns scored highest", safe)):
        lines += [f"## {title}", ""]
        for r in items:
            row = rows_by_index[r["index"]]
            src = row["source"]
            msgs = row["prompt"]
            lines.append(f"### p={r['p_unsafe']:.3f} — {src['dialogue_id']} "
                         f"turn {src['user_turn_index']} (judge: {r['label']})")
            if len(msgs) > 1:
                lines.append(f"> **{msgs[-2]['role']}:** {msgs[-2]['content']}")
                lines.append(">")
            lines.append(f"> **user (assessed):** {msgs[-1]['content']}")
            lines.append("")
    out = Path(run) / f"{model}.misses.md"
    out.write_text("\n".join(lines))
    return out


def _f(x):
    return "  n/a " if x != x else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="score_decisions.py --outdir for one spec version (repeatable)")
    ap.add_argument("--rows", default=DEFAULT_ROWS)
    ap.add_argument("--misses", type=int, default=0,
                    help="write the N worst misses per model to <run>/<model>.misses.md")
    args = ap.parse_args()

    rows_by_index = {r["index"]: r for r in _read_jsonl(args.rows)}
    print(f"{'spec run':14} {'sha':12} {'model / question':24} {'n':>5} {'AUROC':>7} "
          f"{'SH':>7} {'HtO':>7} {'FPR@95':>7}  HtO turns chosen as")
    for run in args.run:
        for model in MODELS:
            if not (Path(run) / f"{model}__yesno_scores.jsonl").exists():
                print(f"{Path(run).name:14} {'':12} {model:24} (not scored)")
                continue
            sha, stats = _spec_sha(run, model)
            if stats["unanswered"]:
                print(f"  ! {model} on {run}: {stats['unanswered']} rows unanswered")
            for q in QUESTIONS:
                s = summarize(run, model, q)
                choices = s.get("hto_choices", "")
                print(f"{Path(run).name:14} {sha:12} {model + ' / ' + q:24} {s['n']:>5} "
                      f"{_f(s['auroc']):>7} {_f(s['sh']):>7} {_f(s['hto']):>7} "
                      f"{_f(s['fpr95']):>7}  {choices}")
            if args.misses:
                print(f"{'':14} wrote {write_misses(run, model, rows_by_index, args.misses)}")


if __name__ == "__main__":
    main()
