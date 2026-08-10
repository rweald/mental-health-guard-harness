# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Author a draft scenario library with a strong reasoning model, for clinical review.

We lack a clinical team to hand-build the ~300 scenarios (paper §4), so a top reasoning
model (Opus 4.8 on high) drafts them. This code owns the distribution — it computes an
exact quota of category × subcategory × trajectory cells from the paper's balance,
over-weighting borderline subcategories — and the model only authors content per cell.

This is a PAID call to a strong model. Per the project guardrails the assistant prepares
this command; you run it. The authoring PROMPT is reviewed and approved before any run
(see mh_guard_harness.authoring.build_author_prompt). Needs $OPENROUTER_API_KEY.

    uv run python scripts/generate/author_scenarios.py --total 300

Writes a DRAFT library (default data/scenarios/library_draft.json). It is a draft for
clinical review, not a finished artifact — hand it to the clinical team before use.
"""

import argparse
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mh_guard_harness.authoring import (
    BatchResult,
    author_batch,
    build_cells,
    cell_summary,
)
from mh_guard_harness.io_utils import ensure_shared_dir, open_shared
from mh_guard_harness.models import OpenRouterModel
from mh_guard_harness.scenarios import RISK_CATEGORIES, to_dict, validate_library

# The authoring model. VERIFY the slug against the OpenRouter catalog before spending —
# an unpinned reasoning model is exactly where a wrong id silently wastes a big run.
MODEL_DEFAULT = "anthropic/claude-opus-4.8"
OUTFILE = Path("data/scenarios/library_draft.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=300, help="scenarios to author")
    ap.add_argument("--batch-size", type=int, default=6, help="scenarios per model call")
    ap.add_argument("--workers", type=int, default=4, help="concurrent authoring calls")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--reasoning-effort", default="high",
                    help="reasoning effort for the authoring model")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--only-category", choices=RISK_CATEGORIES, default=None,
                    help="author only this risk category (targeted debugging / top-up)")
    ap.add_argument("--out", default=str(OUTFILE))
    args = ap.parse_args()

    target = {args.only_category: 1.0} if args.only_category else None
    cells = build_cells(args.total, **({"category_target": target} if target else {}))
    summ = cell_summary(cells)
    print(f"quota: {summ['n']} cells, balance {summ['balance']}", flush=True)

    opts = {"temperature": args.temperature, "reasoning_effort": args.reasoning_effort}
    model = OpenRouterModel(model=args.model, default_opts=opts, timeout=600.0)

    batches = [cells[i : i + args.batch_size] for i in range(0, len(cells), args.batch_size)]
    total_b = len(batches)
    print(f"authoring {args.total} scenarios in {total_b} batches of ≤{args.batch_size} "
          f"with {args.model} (reasoning={args.reasoning_effort})", flush=True)

    lock = threading.Lock()
    done = [0]

    def _run(item):
        bi, batch = item
        try:
            res = author_batch(model, batch, batch_index=bi)
        except Exception as e:  # keep the sweep alive if one batch call fails outright
            res = BatchResult(scenarios=[], raw="",
                              rejects=[{"reason": f"batch call failed: {e}"}])
        with lock:
            done[0] += 1
            print(f"[author] batch {done[0]}/{total_b} -> {len(res.scenarios)}/{len(batch)} "
                  f"valid, {len(res.rejects)} rejected", flush=True)
        return bi, res

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(_run, list(enumerate(batches))))

    drafted = [s for _, res in results for s in res.scenarios]
    rejects = [{"batch_index": bi, **r} for bi, res in results for r in res.rejects]

    # Assign final, readable, unique ids (category + subcategory + running index).
    counter: Counter = Counter()
    for s in drafted:
        key = f"{s.risk_category}__{s.subcategory}"
        s.id = f"{key}__{counter[key]:03d}"
        counter[key] += 1

    report = validate_library(drafted)  # raises if anything slipped through invalid
    print(f"authored {report['n']}/{args.total} valid scenarios; "
          f"balance {report['balance']} (drift {report['drift']})", flush=True)

    ensure_shared_dir(Path(args.out).parent)
    with open_shared(args.out) as f:
        f.write(json.dumps([to_dict(s) for s in drafted], indent=2))
    print(f"wrote DRAFT {args.out} — review with the clinical team before use", flush=True)

    # Loudly surface drops (and dump the raw replies + reasons) so a run that quietly
    # loses whole categories can't pass as a success.
    if rejects:
        by_reason = Counter(r["reason"].split(":")[0] for r in rejects)
        rej_path = Path(args.out).with_suffix(".rejects.json")
        with open_shared(str(rej_path)) as f:
            f.write(json.dumps({
                "rejects": rejects,
                "raw_by_batch": {bi: res.raw for bi, res in results if res.rejects},
            }, indent=2))
        print(f"\n!! {len(rejects)} cell(s) produced no valid scenario "
              f"({dict(by_reason)}). Raw replies + reasons in {rej_path}", flush=True)


if __name__ == "__main__":
    main()
