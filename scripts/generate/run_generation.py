# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Generate the synthetic patient↔clinician dialogue set. Runs the
two-agent loop over a scenario library × seeds and writes one shared dialogue set that
every judge arm later labels.

These are PAID PLM + CLM API calls (OpenRouter). Per the project guardrails the
assistant does not run this — it prepares the command; you run it. Needs
$OPENROUTER_API_KEY.

    uv run python scripts/generate/run_generation.py \
        --scenarios data/scenarios/examples.json \
        --seeds 1

Writes results/generated/dialogues.jsonl (one JSON dialogue per line, keyed by
scenario id + seed). Cost per dialogue ≈ (2 * max_turns - 1) calls split across PLM/CLM.
"""

import argparse
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mh_guard_harness.generation import dialogue_to_dict, generate_dialogue
from mh_guard_harness.io_utils import ensure_shared_dir, open_shared
from mh_guard_harness.models import OpenRouterModel
from mh_guard_harness.scenarios import load_library, validate_library

# Model roles are pinned and held fixed across all judge arms (the judge is the only
# experimental variable). PLM = GPT-5.6 (a deliberate deviation from the paper's GLM-4.6
# patient: it keeps the patient in a different lineage from every judge, so no judge
# grades its own prose). CLM = Claude Sonnet 4.6, the scenario-blind clinician. Override
# with --plm-model / --clm-model only for smoke plumbing, not real runs.
PLM_DEFAULT = "openai/gpt-5.6-luna"
CLM_DEFAULT = "anthropic/claude-sonnet-4.6"
OUTFILE = Path("results/generated/dialogues.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True, help="scenario library JSON")
    ap.add_argument("--clm-model", default=CLM_DEFAULT, help="OpenRouter id for the clinician")
    ap.add_argument("--plm-model", default=PLM_DEFAULT, help="OpenRouter id for the patient")
    ap.add_argument("--seeds", type=int, default=1, help="dialogues per scenario (seeds 0..k-1)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent dialogues")
    ap.add_argument("--limit", type=int, default=None, help="first N scenarios only")
    ap.add_argument("--temperature", type=float, default=0.9, help="patient sampling temp")
    ap.add_argument("--final-clinician-reply", action="store_true",
                    help="also generate a clinician reply after the last patient turn")
    ap.add_argument("--out", default=str(OUTFILE))
    args = ap.parse_args()

    scenarios = load_library(args.scenarios)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    summary = validate_library(scenarios)  # raises on any invalid / duplicate scenario
    print(f"library: {summary['n']} scenarios, balance {summary['balance']}, "
          f"drift-from-paper {summary['drift']}", flush=True)

    plm = OpenRouterModel(model=args.plm_model, default_opts={"temperature": args.temperature})
    clm = OpenRouterModel(model=args.clm_model, default_opts={"temperature": 0.7})

    jobs = [(s, seed) for s in scenarios for seed in range(args.seeds)]
    total = len(jobs)
    est_calls = sum(2 * s.max_turns - (0 if args.final_clinician_reply else 1)
                    for s, _ in jobs)
    print(f"generating {total} dialogues (~{est_calls} API calls) "
          f"PLM={args.plm_model} CLM={args.clm_model}", flush=True)

    lock = threading.Lock()
    done = [0]
    failures: list[dict] = []

    def _run(job):
        scenario, seed = job
        try:
            d = generate_dialogue(
                scenario, plm, clm, seed=seed,
                final_clinician_reply=args.final_clinician_reply,
            )
        except Exception as e:  # keep the sweep alive if one dialogue fails
            with lock:
                done[0] += 1
                failures.append({
                    "scenario_id": scenario.id, "seed": seed,
                    "risk_category": scenario.risk_category,
                    "error": f"{type(e).__name__}: {e}",
                })
                print(f"[gen] {done[0]}/{total} {scenario.id}#seed{seed} FAILED: {e}",
                      flush=True)
            return None
        with lock:
            done[0] += 1
            note = f" [{len(d.flags)} flagged]" if d.flags else ""
            print(f"[gen] {done[0]}/{total} {d.key()} ({len(d.user_turns())} turns){note}",
                  flush=True)
        return d

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        dialogues = [d for d in ex.map(_run, jobs) if d is not None]

    ensure_shared_dir(Path(args.out).parent)
    with open_shared(args.out) as f:
        for d in dialogues:
            f.write(json.dumps(dialogue_to_dict(d)) + "\n")
    print(f"wrote {args.out} ({len(dialogues)}/{total} dialogues)", flush=True)

    # Loudly surface systematic loss: a scattered per-line FAILED can hide a whole risk
    # category vanishing (e.g. a provider filter). Summarize by category + dump for re-run.
    if failures:
        by_cat = Counter(f["risk_category"] for f in failures)
        fpath = Path(args.out).with_suffix(".failures.json")
        with open_shared(str(fpath)) as f:
            f.write(json.dumps(failures, indent=2))
        print(f"\n!! {len(failures)}/{total} dialogues FAILED (by category {dict(by_cat)}). "
              f"Details in {fpath}", flush=True)

    # Non-fatal quality concerns (turns that read as refusals): kept, but flagged for a
    # human to review before these dialogues are labeled and trained on.
    flagged = [d for d in dialogues if d.flags]
    if flagged:
        gpath = Path(args.out).with_suffix(".flagged.json")
        payload = [{"dialogue_id": d.key(), "scenario_id": d.scenario_id,
                    "seed": d.seed, "flags": d.flags} for d in flagged]
        with open_shared(str(gpath)) as f:
            f.write(json.dumps(payload, indent=2))
        print(f"\n!! {len(flagged)} dialogue(s) flagged for review (possible refusals). "
              f"Details in {gpath}", flush=True)


if __name__ == "__main__":
    main()
