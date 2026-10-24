# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Compare decision models against MindGuard-8B and the GLM-5.2 fine-tune. Torch-free.

Inputs are all seam-contract scores files over the same testset rows:

  - one per decision model and question (``<model>__choice3`` / ``<model>__yesno``),
    plus that model's failures file and provenance, from ``score_decisions.py``;
  - the MindGuard-8B baseline (one file);
  - the GLM-5.2 fine-tune (one file per seed).

Every result is reported under two failure policies. **excluded** scores only the rows
the model answered, and pairs baselines on those same rows. **as_unsafe** keeps every
row and scores a failed one as ``p_unsafe = 1.0`` — the fail-closed reading, where a
refusal to judge a turn is treated as flagging it. When a model has no failures the two
are identical and only computed once.

Uncertainty is a paired, stratified bootstrap over testset rows (``analysis.
bootstrap_auroc_diff``), overall and per risk category. With ~90 comparisons on 42
positives some raw p-values will dip under 0.05 by chance, so each comes with a
Holm-adjusted value across the whole report; read significance from that one.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .analysis import bootstrap_auroc_diff

QUESTIONS = ("choice3", "yesno")
SUBGROUPS = (metrics.SELF_HARM, metrics.HARM_TO_OTHERS)
SCOPES = ("overall", *SUBGROUPS)
POLICIES = ("excluded", "as_unsafe")
CATEGORIES = (metrics.SAFE, *SUBGROUPS)

# Display order; any other model found on disk is appended after these.
MODEL_ORDER = ("jev-1.14", "clef", "clef-flash", "gpt-6-luna", "decider-v1.2", "decision-1")

MINDGUARD = "MindGuard-8B"
GLM = "GLM-5.2 fine-tune"


@dataclass
class Entry:
    """One scored system: per-row scores aligned to the testset index."""

    name: str
    p_unsafe: pd.Series                  # index -> score, scored rows only
    failed: list[int] = field(default_factory=list)
    choice: pd.Series | None = None      # index -> chosen category (choice3 only)
    provenance: dict | None = None
    seeds: list[pd.Series] | None = None  # per-seed scores (GLM fine-tune only)


def _scores(path) -> tuple[pd.DataFrame, pd.Series]:
    df = metrics.load_scores(path).set_index("index", drop=False)
    if df.index.has_duplicates:
        raise ValueError(f"{path}: duplicate row indexes")
    return df, df["p_unsafe"]


def load_gold(path) -> pd.DataFrame:
    """Gold labels by index, from any complete scores file over the testset."""
    df, _ = _scores(path)
    return df[["label", "y_true", "risk_category"]]


def check_aligned(gold: pd.DataFrame, path) -> None:
    """Every row in ``path`` must exist in gold with the same label — otherwise the
    pairing that the bootstrap depends on is wrong."""
    df, _ = _scores(path)
    missing = df.index.difference(gold.index)
    if len(missing):
        raise ValueError(f"{path}: {len(missing)} row indexes not in the gold set")
    bad = (df["label"] != gold.loc[df.index, "label"]).sum()
    if bad:
        raise ValueError(f"{path}: {bad} rows disagree with gold on the label")


def load_mindguard(path, gold) -> Entry:
    check_aligned(gold, path)
    _, p = _scores(path)
    _require_complete(path, p, gold)
    return Entry(MINDGUARD, p)


def load_glm(paths, gold) -> Entry:
    """Per-seed scores, plus their per-row mean as the bootstrap input (averaging
    collapses training noise, so the interval reflects testset sampling only)."""
    seeds = []
    for path in paths:
        check_aligned(gold, path)
        _, p = _scores(path)
        _require_complete(path, p, gold)
        seeds.append(p.sort_index())
    mean = pd.concat(seeds, axis=1).mean(axis=1)
    return Entry(GLM, mean, seeds=seeds)


def _require_complete(path, p, gold):
    if len(p) != len(gold):
        raise ValueError(f"{path}: {len(p)} rows, expected {len(gold)}")


def find_models(decisions_dir) -> list[str]:
    """Model keys with a full (non ``--limit``) choice3 scores file on disk."""
    found = sorted(
        p.name.removesuffix("__choice3_scores.jsonl")
        for p in Path(decisions_dir).glob("*__choice3_scores.jsonl")
        if "__limit" not in p.name
    )
    return [m for m in MODEL_ORDER if m in found] + [m for m in found if m not in MODEL_ORDER]


def load_decision(decisions_dir, model: str, question: str, gold) -> Entry:
    d = Path(decisions_dir)
    path = d / f"{model}__{question}_scores.jsonl"
    check_aligned(gold, path)
    df, p = _scores(path)
    prov = json.loads(path.with_suffix(".provenance.json").read_text())
    q_name = prov["question"]
    failed = []
    failures = d / f"{model}.failures.jsonl"
    if failures.exists():
        for line in failures.read_text().splitlines():
            if line.strip():
                f = json.loads(line)
                if f["question"] == q_name:
                    failed.append(f["index"])
    scored, listed = set(p.index), set(failed)
    if scored & listed or (scored | listed) != set(gold.index):
        raise ValueError(
            f"{model}/{question}: scored rows and failures don't partition the testset "
            f"({len(scored)} scored + {len(listed)} failed vs {len(gold)} rows) — rerun "
            "score_decisions.py for this model on the full rows file"
        )
    return Entry(f"{model} / {question}", p, failed=sorted(failed),
                 choice=_choices(path), provenance=prov)


def _choices(path) -> pd.Series | None:
    """The three-way answer's chosen option per row. ``metrics.load_scores`` keeps only
    the seam fields, so this reads the extra field straight from the file."""
    picks = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if "choice" in rec:
                picks[rec["index"]] = rec["choice"]
    return pd.Series(picks) if picks else None


def scores_under(entry: Entry, policy: str, gold) -> pd.Series:
    """The entry's scores over the rows a policy keeps, in index order."""
    if policy == "as_unsafe" and entry.failed:
        filled = pd.concat([entry.p_unsafe, pd.Series(1.0, index=entry.failed)])
        return filled.sort_index()
    return entry.p_unsafe.sort_index()


def _metrics(p: pd.Series, gold) -> dict:
    g = gold.loc[p.index]
    y, cats, s = g["y_true"].to_numpy(), g["risk_category"].to_numpy(), p.to_numpy()
    at = metrics.fpr_at_90_95(y, s)
    return {
        "n": int(len(p)),
        "n_positive": int(y.sum()),
        "auroc": metrics.auroc(y, s),
        "fpr_at_90": at["fpr_at_90"]["fpr"],
        "fpr_at_95": at["fpr_at_95"]["fpr"],
        "subgroup_auroc": {sg: metrics.subgroup_auroc(cats, s, sg) for sg in SUBGROUPS},
        # Exact, not rounded: a model that returns two-decimal probabilities shows up
        # here as a handful of values, and that coarseness is part of its result.
        "distinct_scores": int(np.unique(s).size),
        "pinned_at_0_or_1": float(((s <= 1e-6) | (s >= 1 - 1e-6)).mean()),
    }


def confusion(entry: Entry, gold) -> dict | None:
    """Gold category -> chosen category counts, for the three-way question."""
    if entry.choice is None:
        return None
    cats = gold.loc[entry.choice.index, "risk_category"]
    table = pd.crosstab(cats, entry.choice).reindex(
        index=list(CATEGORIES), columns=list(CATEGORIES), fill_value=0
    )
    return {g: {c: int(table.loc[g, c]) for c in CATEGORIES} for g in CATEGORIES}


def _run_stats(prov: dict) -> dict:
    run = prov["run"]
    answered = run["answered"] or 1
    return {
        "cost_usd": run["cost_usd"],
        "cost_per_1k_turns_usd": run["cost_usd"] / answered * 1000,
        "latency_p50_s": run["latency_s"]["p50"],
        "latency_p95_s": run["latency_s"]["p95"],
        "resolved_models": run["resolved_models"],
        "providers": run["providers"],
        "rows_over_long_state_bytes": run["rows_over_long_state_bytes"],
        "failures": prov["failures"],
        "failures_by_class": prov.get("failures_by_class", {}),
    }


# --- bootstrap ------------------------------------------------------------------------


def _paired(a: pd.Series, b: pd.Series, gold, scope: str):
    """Arrays for one paired comparison: rows both sides scored, narrowed to the scope."""
    rows = a.index.intersection(b.index)
    cats = gold.loc[rows, "risk_category"].to_numpy()
    if scope == "overall":
        keep = np.ones(len(rows), dtype=bool)
        y = gold.loc[rows, "y_true"].to_numpy()
    else:
        keep = (cats == scope) | (cats == metrics.SAFE)
        y = (cats == scope).astype(int)
    return y[keep], a.loc[rows].to_numpy()[keep], b.loc[rows].to_numpy()[keep], int(keep.sum())


def _boot(job):
    y, pa, pb, n_boot, seed = job
    return bootstrap_auroc_diff(y, pa, pb, n_boot=n_boot, seed=seed)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order. NaNs pass through and
    don't count toward the family size."""
    p = np.asarray(pvalues, dtype=float)
    ok = np.flatnonzero(~np.isnan(p))
    out = np.full(p.shape, np.nan)
    m = len(ok)
    order = ok[np.argsort(p[ok], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


# --- report ---------------------------------------------------------------------------


def build(
    *,
    decisions_dir,
    mindguard_path,
    glm_paths,
    n_boot: int = 10_000,
    seed: int = 0,
    workers: int | None = None,
) -> dict:
    gold = load_gold(mindguard_path)
    mindguard = load_mindguard(mindguard_path, gold)
    glm = load_glm(glm_paths, gold)
    models = find_models(decisions_dir)
    if not models:
        raise SystemExit(f"no decision-model scores under {decisions_dir}")

    entries = {m: {q: load_decision(decisions_dir, m, q, gold) for q in QUESTIONS}
               for m in models}
    specs = {e.provenance["decision_spec"]["sha256"]
             for qs in entries.values() for e in qs.values()}
    if len(specs) != 1:
        raise ValueError(f"decision models were scored under {len(specs)} different "
                         "specs; a comparison needs one")

    report = {
        "n_rows": len(gold),
        "n_positive": int(gold["y_true"].sum()),
        "decision_spec_sha256": specs.pop(),
        "baselines": {
            MINDGUARD: _metrics(mindguard.p_unsafe.sort_index(), gold),
            GLM: {
                "seed_averaged": _metrics(glm.p_unsafe, gold),
                "per_seed_auroc": [metrics.auroc(gold.loc[s.index, "y_true"], s)
                                   for s in glm.seeds],
            },
        },
        "models": {},
        "comparisons": [],
        "n_boot": n_boot,
    }
    g = report["baselines"][GLM]
    a = np.asarray(g["per_seed_auroc"])
    g["per_seed_auroc_mean"], g["per_seed_auroc_sd"] = float(a.mean()), float(a.std(ddof=1))

    jobs, labels = [], []

    def compare(name_a, pa, name_b, pb, policy):
        for scope in SCOPES:
            y, xa, xb, n = _paired(pa, pb, gold, scope)
            jobs.append((y, xa, xb, n_boot, seed))
            labels.append({"a": name_a, "b": name_b, "policy": policy, "scope": scope,
                           "n": n})

    for m, qs in entries.items():
        # Metrics are shown under both policies for every model so the tables line up;
        # the bootstrap runs a second policy only where it differs from the first.
        policies = POLICIES if any(e.failed for e in qs.values()) else ("as_unsafe",)
        report["models"][m] = {
            "bootstrap_policies": list(policies),
            "questions": {},
            "run": _run_stats(qs["choice3"].provenance),
        }
        for q, e in qs.items():
            report["models"][m]["questions"][q] = {
                "metrics": {pol: _metrics(scores_under(e, pol, gold), gold)
                            for pol in POLICIES},
                "failed_rows": len(e.failed),
                "confusion": confusion(e, gold),
            }
        for pol in policies:
            for q, e in qs.items():
                p = scores_under(e, pol, gold)
                compare(e.name, p, MINDGUARD, mindguard.p_unsafe, pol)
                compare(e.name, p, GLM, glm.p_unsafe, pol)
            compare(qs["choice3"].name, scores_under(qs["choice3"], pol, gold),
                    qs["yesno"].name, scores_under(qs["yesno"], pol, gold), pol)

    if workers == 1:
        results = [_boot(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_boot, jobs, chunksize=1))
    adjusted = holm([r["pvalue"] for r in results])
    for label, r, p_holm in zip(labels, results, adjusted):
        report["comparisons"].append({**label, **r, "p_holm": p_holm})
    return report


# --- text rendering -------------------------------------------------------------------


def _f(x, fmt=".4f"):
    return "  n/a " if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)


def render(report: dict) -> str:
    out = []
    w = out.append
    w(f"Testset: {report['n_rows']} rows, {report['n_positive']} positives. "
      f"Decision spec sha256 {report['decision_spec_sha256'][:12]}.")
    w("")
    w("=== Baselines ===")
    mg = report["baselines"][MINDGUARD]
    glm = report["baselines"][GLM]
    ga = glm["seed_averaged"]
    for name, x in ((MINDGUARD, mg), (GLM + " (seed avg)", ga)):
        sub = x["subgroup_auroc"]
        w(f"  {name:28} AUROC {_f(x['auroc'])}  FPR@90 {_f(x['fpr_at_90'])}  "
          f"FPR@95 {_f(x['fpr_at_95'])}  SH {_f(sub['self_harm'])}  "
          f"HtO {_f(sub['harm_to_others'])}  distinct {x['distinct_scores']}")
    w(f"  {'':28} per-seed AUROC {glm['per_seed_auroc_mean']:.4f} ± "
      f"{glm['per_seed_auroc_sd']:.4f} (training variance)")

    for policy in POLICIES:
        no_failures = all(d["questions"][q]["failed_rows"] == 0
                          for d in report["models"].values() for q in QUESTIONS)
        if no_failures and policy == "excluded":
            continue  # identical to the as_unsafe table
        rows = [(m, q, d["questions"][q]["metrics"][policy], d["questions"][q]["failed_rows"])
                for m, d in report["models"].items() for q in QUESTIONS]
        title = {"excluded": "failures excluded",
                 "as_unsafe": "failures scored as unsafe"}[policy]
        if no_failures:
            title = "no model had failures, so both policies coincide"
        w("")
        w(f"=== Decision models ({title}) ===")
        w(f"  {'model / question':26} {'n':>5} {'AUROC':>7} {'FPR@90':>7} {'FPR@95':>7} "
          f"{'SH AUROC':>8} {'HtO AUROC':>9} {'distinct':>8} {'pinned':>7} {'failed':>6}")
        for m, q, x, nf in rows:
            sub = x["subgroup_auroc"]
            w(f"  {m + ' / ' + q:26} {x['n']:>5} {_f(x['auroc']):>7} {_f(x['fpr_at_90']):>7} "
              f"{_f(x['fpr_at_95']):>7} {_f(sub['self_harm']):>8} "
              f"{_f(sub['harm_to_others']):>9} {x['distinct_scores']:>8} "
              f"{x['pinned_at_0_or_1']:>7.1%} {nf:>6}")

    w("")
    w("=== Cost and latency ===")
    w(f"  {'model':14} {'$ total':>9} {'$ / 1k turns':>12} {'p50 s':>7} {'p95 s':>7}  served as")
    for m, d in report["models"].items():
        r = d["run"]
        served = ", ".join(f"{k} via {p}" for k, p in zip(r["resolved_models"], r["providers"]))
        w(f"  {m:14} {r['cost_usd']:>9.4f} {r['cost_per_1k_turns_usd']:>12.4f} "
          f"{_f(r['latency_p50_s'], '.3f'):>7} {_f(r['latency_p95_s'], '.3f'):>7}  {served}")

    w("")
    w("=== Three-way confusion (gold rows x chosen column; choice3 question) ===")
    for m, d in report["models"].items():
        c = d["questions"]["choice3"]["confusion"]
        if not c:
            continue
        w(f"  {m}")
        w("    " + "gold / chose".ljust(16) + "".join(f"{x:>16}" for x in CATEGORIES))
        for g in CATEGORIES:
            w(f"    {g:16}" + "".join(f"{c[g][x]:>16}" for x in CATEGORIES))

    w("")
    w(f"=== Paired bootstrap, AUROC(a) - AUROC(b) ({report['n_boot']} resamples) ===")
    w("  * = Holm-adjusted p < 0.05 across every comparison in this report")
    for c in report["comparisons"]:
        star = " *" if c["p_holm"] < 0.05 else ""
        w(f"  [{c['policy']:9}] {c['a']:26} vs {c['b']:22} {c['scope']:15} "
          f"Δ {_f(c['delta'], '+.4f')}  CI [{_f(c['ci_low'], '+.4f')}, "
          f"{_f(c['ci_high'], '+.4f')}]  p {_f(c['pvalue'])}  holm {_f(c['p_holm'])}{star}")
    return "\n".join(out)
