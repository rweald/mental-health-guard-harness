# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Audit a generated dialogue set before spending on labeling — between generation and labeling.

Prints a status table (coverage, structural integrity, output hygiene, degeneracy) and
exits non-zero if any check fails — so it doubles as a gate you can run by hand or wire
into a pre-labeling check. Pure analysis, no API calls.

    uv run python scripts/generate/audit_dialogues.py \
        --dialogues results/generated/dialogues.jsonl \
        --scenarios data/scenarios/library.json

Pass --json to also dump the machine-readable detail counts.
"""

import argparse
import json
import sys
from pathlib import Path

from mh_guard_harness.dialogue_audit import audit_dialogues, format_report
from mh_guard_harness.generation import load_dialogues
from mh_guard_harness.scenarios import load_library


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", default="results/generated/dialogues.jsonl")
    ap.add_argument("--scenarios", default="data/scenarios/library.json",
                    help="scenario library the set was generated from (enables coverage "
                         "/ turn-count / category checks); pass '' to skip")
    ap.add_argument("--json", action="store_true", help="also print detail counts as JSON")
    args = ap.parse_args()

    dialogues = load_dialogues(args.dialogues)
    scenarios = load_library(args.scenarios) if args.scenarios and Path(args.scenarios).exists() else None

    report = audit_dialogues(dialogues, scenarios)
    print(format_report(report, title=f"dialogue audit — {args.dialogues}"))
    if args.json:
        print(json.dumps(report.details, indent=2))

    sys.exit(1 if report.failures else 0)


if __name__ == "__main__":
    main()
