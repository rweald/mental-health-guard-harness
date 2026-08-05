# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Audit a judge's label set against the dialogues it labeled — between labeling and fine-tuning.

Prints a status table — integrity (coverage, nulls, duplicates) plus judge-quality
signals (tag→label confusion, UNSAFE recall, unsafe-type match, safe-scenario FPR) — and
exits non-zero only on an integrity *failure*. Judge-quality signals are reported, never
failed (a weak judge is a valid result). Pure analysis, no API calls.

    uv run python scripts/generate/audit_labels.py \
        --labels results/generated/labels/glm-5.2__full.jsonl

Run it once per judge to get comparable tables for the judge comparison. Pass --json to
also dump the machine-readable detail counts.
"""

import argparse
import json
import sys
from pathlib import Path

from mh_guard_harness.generation import load_dialogues
from mh_guard_harness.label_audit import audit_labels, format_report, load_label_rows
from mh_guard_harness.scenarios import load_library


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="a <judge>__<mode>.jsonl label file")
    ap.add_argument("--dialogues", default="results/generated/dialogues.jsonl")
    ap.add_argument("--scenarios", default="data/scenarios/library.json",
                    help="scenario library (enables unsafe-type / safe-FPR signals); "
                         "pass '' to skip")
    ap.add_argument("--json", action="store_true", help="also print detail counts as JSON")
    args = ap.parse_args()

    rows = load_label_rows(args.labels)
    dialogues = load_dialogues(args.dialogues)
    scenarios = load_library(args.scenarios) if args.scenarios and Path(args.scenarios).exists() else None

    report = audit_labels(rows, dialogues, scenarios)
    print(format_report(report, title=f"label audit — {args.labels}"))
    if args.json:
        print(json.dumps(report.details, indent=2))

    sys.exit(1 if report.failures else 0)


if __name__ == "__main__":
    main()
