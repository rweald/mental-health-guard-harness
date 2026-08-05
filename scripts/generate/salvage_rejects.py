# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Salvage a scenario-authoring draft: recover the rejects that failed only to parser
brittleness (trailing commas, one malformed sibling) or the since-relaxed safe-turn
rule, WITHOUT re-spending on the model. It re-processes the raw replies already captured
in the run's ``*.rejects.json``.

No API calls — pure re-parse + re-validate of saved output. Merges the recovered
scenarios into the valid draft, re-assigns final readable ids, and writes a fuller
library plus a residual reject report for whatever genuinely needs re-authoring.

    uv run python scripts/generate/salvage_rejects.py \
        --draft data/scenarios/library_draft.json \
        --out data/scenarios/library.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from mh_guard_harness.authoring import recover_rejects
from mh_guard_harness.io_utils import ensure_shared_dir, open_shared
from mh_guard_harness.scenarios import (
    from_dict,
    to_dict,
    validate_library,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="data/scenarios/library_draft.json",
                    help="the valid scenarios the run already wrote")
    ap.add_argument("--rejects", default=None,
                    help="the run's rejects file (default: <draft>.rejects.json)")
    ap.add_argument("--out", default="data/scenarios/library.json",
                    help="merged, re-id'd library to write")
    args = ap.parse_args()

    draft_path = Path(args.draft)
    rejects_path = Path(args.rejects) if args.rejects else draft_path.with_suffix(".rejects.json")

    kept = [from_dict(d) for d in json.loads(draft_path.read_text())]
    payload = json.loads(rejects_path.read_text())
    n_rejects = len(payload.get("rejects", []))
    recovered, still = recover_rejects(payload)
    print(f"draft: {len(kept)} valid; rejects: {n_rejects} -> "
          f"recovered {len(recovered)}, still {len(still)}", flush=True)

    merged = kept + recovered

    # Assign final, readable, unique ids (category + subcategory + running index), same
    # scheme as the authoring runner, so a salvaged library is indistinguishable from a
    # clean one downstream.
    counter: Counter = Counter()
    for s in merged:
        key = f"{s.risk_category}__{s.subcategory}"
        s.id = f"{key}__{counter[key]:03d}"
        counter[key] += 1

    report = validate_library(merged)  # raises if any recovered scenario slipped through
    print(f"merged {report['n']} scenarios; balance {report['balance']} "
          f"(drift {report['drift']})", flush=True)

    ensure_shared_dir(Path(args.out).parent)
    with open_shared(args.out) as f:
        f.write(json.dumps([to_dict(s) for s in merged], indent=2))
    print(f"wrote {args.out} — still a DRAFT for clinical review", flush=True)

    if still:
        by_reason = Counter(
            (r.get("recover_reason") or r.get("reason", "")).split(":")[0] for r in still
        )
        residual = Path(args.out).with_suffix(".residual_rejects.json")
        with open_shared(str(residual)) as f:
            f.write(json.dumps(still, indent=2))
        print(f"\n!! {len(still)} cell(s) still need re-authoring "
              f"({dict(by_reason)}). Details in {residual}", flush=True)


if __name__ == "__main__":
    main()
