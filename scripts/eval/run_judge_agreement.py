# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Step: judge-agreement (local). Run each judge over the reconstructed conversations
in both context modes and write per-turn labels for the agreement analysis.

Judges are OpenRouter API calls (no GPU), so this runs in the local env and needs
$OPENROUTER_API_KEY. Reads results/eval/testset_conversations.json (from
extract_conversations.py); writes results/eval/judge_labels/<judge>__<mode>.jsonl.

    uv run python scripts/eval/run_judge_agreement.py --limit 3            # smoke
    uv run python scripts/eval/run_judge_agreement.py --judges glm-4.6 --modes full

Cost: 'full' = n_samples calls per conversation (67 total); 'turn' = n_samples calls
per gold turn (1134 total). Multiply by #judges x #modes.
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mh_guard_harness import paper_prompts
from mh_guard_harness.conversations import from_dicts
from mh_guard_harness.io_utils import ensure_shared_dir, open_shared
from mh_guard_harness.judging import MODE_FULL, MODE_TURN, label_conversation
from mh_guard_harness.models import OpenRouterModel

# OpenRouter model ids per judge. VERIFY against the OpenRouter catalog before a real
# run — these gate which model actually answers. Override with --model-id if needed.
JUDGES = {
    "glm-4.6": {"model": "z-ai/glm-4.6", "opts": {"temperature": 0.7}},
    "glm-5.2": {"model": "z-ai/glm-5.2", "opts": {"temperature": 0.7}},
    "deepseek-v4-flash": {"model": "deepseek/deepseek-v4-flash", "opts": {"temperature": 0.7}},
}

CONVS = Path("results/eval/testset_conversations.json")
OUTDIR = Path("results/eval/judge_labels")


class _ProgressModel:
    """Wraps a Model to print a one-line tick as each API call completes, so a long
    sweep is visibly alive rather than looking hung. Lives here (not in the core
    adapter) so models.py / judging.py stay unchanged."""

    def __init__(self, inner, total, label):
        self._inner = inner
        self._total = total
        self._label = label
        self._n = 0
        self._lock = threading.Lock()  # ticks come from many worker threads

    def complete(self, messages, **opts):
        # Tick as each call dispatches. Hold the lock only for the counter+print, not
        # the network call, so workers don't serialize on it.
        with self._lock:
            self._n += 1
            n = self._n
        print(f"[{self._label}] call {n}/{self._total} ...", flush=True)
        return self._inner.complete(messages, **opts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=list(JUDGES), choices=list(JUDGES))
    ap.add_argument("--modes", nargs="+", default=[MODE_FULL, MODE_TURN],
                    choices=[MODE_FULL, MODE_TURN])
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    ap.add_argument("--limit", type=int, default=None, help="first N conversations only")
    ap.add_argument("--conversations", default=str(CONVS))
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--judge-prompt", required=True,
                    help="path to the judge prompt file every judge is run with")
    ap.add_argument("--reproduction-mode", action="store_true",
                    help="require the judge prompt to hash-match the reference prompt")
    args = ap.parse_args()

    paper_prompts.emit_reproduction_notice(args.reproduction_mode)

    # Validate the prompt before spending anything: every judge in the sweep gets this
    # exact prompt, so a bad file would invalidate the whole comparison.
    judge_prompt = paper_prompts.load(
        "judge",
        path=args.judge_prompt,
        mode=paper_prompts.mode_for(args.reproduction_mode),
    )
    print(f"judge prompt: {judge_prompt.filename} "
          f"({judge_prompt.source}, sha256 {judge_prompt.sha256[:12]})", flush=True)

    convs = from_dicts(json.loads(Path(args.conversations).read_text()))
    if args.limit is not None:
        convs = convs[: args.limit]
    ensure_shared_dir(args.outdir)

    for jname in args.judges:
        spec = JUDGES[jname]
        model = OpenRouterModel(model=spec["model"], default_opts=spec["opts"])
        for mode in args.modes:
            out = Path(args.outdir) / f"{jname}__{mode}.jsonl"
            # Total API calls this file will make, so the tick has a denominator.
            total_calls = args.n_samples * (
                len(convs) if mode == MODE_FULL
                else sum(len(c.gold_turns) for c in convs)
            )
            pmodel = _ProgressModel(model, total_calls, f"{jname}/{mode}")

            def _label(conv, mode=mode, jname=jname, pmodel=pmodel):
                gold = {g.user_turn_index: g.label for g in conv.gold_turns}
                try:
                    labels = label_conversation(
                        pmodel, conv, mode=mode, n_samples=args.n_samples,
                        judge_prompt_path=args.judge_prompt,
                        reproduction_mode=args.reproduction_mode,
                    )
                except Exception as e:  # keep the sweep alive if one conversation fails
                    print(f"[{jname}/{mode}] {conv.id} FAILED: {e}", flush=True)
                    labels = {}
                return conv.id, gold, labels

            # Conversations are independent; fan them across workers. ex.map preserves
            # order, so we collect all results then write the file single-threaded.
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(_label, convs))

            n_turns = 0
            with open_shared(out) as f:
                for conv_id, gold, labels in results:
                    for uti, jl in labels.items():
                        f.write(json.dumps({
                            "conv_id": conv_id, "user_turn_index": uti,
                            "judge_label": jl, "gold_label": gold.get(uti),
                        }) + "\n")
                        n_turns += 1
            out.with_suffix(".provenance.json").write_text(json.dumps({
                "judge": jname,
                "model": spec["model"],
                "opts": spec["opts"],
                "context_mode": mode,
                "n_samples": args.n_samples,
                "judge_prompt": paper_prompts.provenance(judge_prompt),
            }, indent=2))
            print(f"[{jname}/{mode}] wrote {out} ({n_turns} turns)", flush=True)


if __name__ == "__main__":
    main()
