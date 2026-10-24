# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Score a decision model over an eval set through OpenRouter. Local, torch-free, PAID.

Sends each row once with both questions from the decision spec and writes two
seam-contract scores files — ``<key>__choice3_scores.jsonl`` and
``<key>__yesno_scores.jsonl`` — so report_metrics.py and the comparison code run on
either unchanged. Raw responses are cached first; a rerun resumes, retrying only rows
that failed, and ``--derive-only`` rebuilds the scores files from the cache offline.

How to run it — export, smoke, full run — is in docs/running-the-decision-eval.md.
"""

import argparse
import json
import sys
from pathlib import Path

from mh_guard_harness import decisions
from mh_guard_harness.models import DecisionAPIError, OpenRouterDecisionModel

# One entry per arm. Every arm denies data collection, since the eval set is sensitive
# and should not end up in anyone's training data. ZDR is on wherever OpenRouter offers
# it — every arm except GPT-6 Luna, whose only endpoint is OpenAI's, which isn't ZDR.
# Clef and Clef-flash are pinned to Cloudflare's own endpoint: the other hosts serve an
# fp8 build or a 16k context, and letting requests fall between them would mix models.
_PRIVATE = {"data_collection": "deny", "allow_fallbacks": False}
MODELS = {
    "jev-1.14": {
        "model": "typesafe/jev-1.14",
        "provider": {**_PRIVATE, "zdr": True, "only": ["typesafe"]},
    },
    "clef": {
        "model": "cloudflare/clef",
        "provider": {**_PRIVATE, "zdr": True, "only": ["cloudflare"]},
    },
    "clef-flash": {
        "model": "cloudflare/clef-flash",
        "provider": {**_PRIVATE, "zdr": True, "only": ["cloudflare"]},
    },
    "gpt-6-luna": {
        "model": "openai/gpt-6-luna-decisions",
        "provider": {**_PRIVATE, "only": ["openai"]},
    },
    "decider-v1.2": {
        "model": "perplexity/pplx-decider-v1.2",
        "provider": {**_PRIVATE, "zdr": True, "only": ["perplexity"]},
    },
    "decision-1": {
        "model": "microsoft/microsoft-decision-1",
        "provider": {**_PRIVATE, "zdr": True, "only": ["azure"]},
    },
}

DEFAULT_ROWS = "results/eval/testset_rows.jsonl"
DEFAULT_OUTDIR = "results/eval/decisions"


def _load_rows(path, limit):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows if limit is None else rows[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--spec", required=True, help="path to the decision spec JSON")
    ap.add_argument("--rows", default=DEFAULT_ROWS)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N rows")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--derive-only", action="store_true",
                    help="rebuild scores files from the response cache; no API calls")
    args = ap.parse_args()

    # Validate the spec before any network call, so a bad file costs nothing.
    spec = decisions.load_spec(args.spec)
    arm = MODELS[args.model]
    rows = _load_rows(args.rows, args.limit)
    decisions.check_rows(rows)
    outdir = Path(args.outdir)
    # The response cache is shared by full and --limit runs (rows match by content), but
    # a --limit run's scores files get their own names so they can't replace the arm's.
    cache_path = outdir / f"{args.model}.responses.jsonl"
    key = args.model if args.limit is None else f"{args.model}__limit{args.limit}"
    print(f"decision spec: {spec.filename} (sha256 {spec.sha256[:12]})", flush=True)
    print(f"{args.model} -> {arm['model']} over {len(rows)} rows from {args.rows}", flush=True)

    if not args.derive_only:
        model = OpenRouterDecisionModel(model=arm["model"], provider=arm["provider"])

        def progress(n, total, entry):
            if not entry["ok"]:
                print(f"  row {entry['index']} FAILED: {entry['error']['message']}",
                      flush=True)
            if n % 50 == 0 or n == total:
                print(f"  {n}/{total} requests", flush=True)

        try:
            made = decisions.run(
                model, rows, spec, cache_path,
                model_id=arm["model"], workers=args.workers, progress=progress,
            )
        except DecisionAPIError as e:  # only fatal ones escape run()
            sys.exit(f"stopped: {e}. Queued rows were not sent; fix it and rerun to resume.")
        except KeyboardInterrupt:
            sys.exit("interrupted: queued rows were not sent; rerun to resume.")
        print(f"made {made} requests (rows already answered were skipped)", flush=True)

    summary = decisions.derive(
        rows, cache_path, outdir,
        key=key, spec=spec, model_id=arm["model"],
        provenance_base={
            "model_key": args.model,
            "model": arm["model"],
            "provider_preferences": arm["provider"],
            "rows_file": Path(args.rows).name,
            "limit": args.limit,
        },
    )
    print(json.dumps(summary, indent=2))
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if summary["run"]["unanswered"]:
        print(f"{summary['run']['unanswered']} rows still unanswered — rerun the same "
              "command to retry them", file=sys.stderr)


if __name__ == "__main__":
    main()
