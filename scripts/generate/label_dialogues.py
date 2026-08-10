# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Label the generated dialogues with a judge to produce the
training labels. Runs the §7.2 judge over each generated dialogue — every user turn is
labeled — and writes per-turn labels that the fine-tune step joins back to the dialogue
text to build training examples.

This is the step that isolates the judge: the SAME dialogue set is labeled once per
judge, so any downstream difference traces to the judge alone. Full mode is the paper's
method (all turns judged with full-conversation context); turn mode is our past-only
variant.

These are PAID judge API calls (OpenRouter). Per the project guardrails the assistant
prepares this command; you run it. Needs $OPENROUTER_API_KEY.

    uv run python scripts/generate/label_dialogues.py \
        --dialogues results/generated/dialogues.jsonl --judges glm-5.2 --modes full

Writes results/generated/labels/<judge>__<mode>.jsonl, one record per labeled turn:
{dialogue_id, user_turn_index, judge_label}.
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mh_guard_harness import paper_prompts
from mh_guard_harness.generation import load_dialogues
from mh_guard_harness.io_utils import ensure_shared_dir, open_shared
from mh_guard_harness.judging import MODE_FULL, MODE_TURN, label_conversation
from mh_guard_harness.models import OpenRouterModel

# Same judge roster as the agreement study (verified: glm-4.6; verify the others before a
# real run). Kept identical so testset agreement and training labels use one judge config.
JUDGES = {
    "glm-4.6": {"model": "z-ai/glm-4.6", "opts": {"temperature": 0.7}},
    "glm-5.2": {"model": "z-ai/glm-5.2", "opts": {"temperature": 0.7}},
    "deepseek-v4-flash": {"model": "deepseek/deepseek-v4-flash", "opts": {"temperature": 0.7}},
}

OUTDIR = Path("results/generated/labels")


class _ProgressModel:
    """Prints a one-line tick per API call so a long sweep is visibly alive. Thread-safe;
    holds the lock only for the counter+print, not the network call."""

    def __init__(self, inner, total, label):
        self._inner, self._total, self._label = inner, total, label
        self._n = 0
        self._lock = threading.Lock()

    def complete(self, messages, **opts):
        with self._lock:
            self._n += 1
            n = self._n
        print(f"[{self._label}] call {n}/{self._total} ...", flush=True)
        return self._inner.complete(messages, **opts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", default="results/generated/dialogues.jsonl")
    ap.add_argument("--judges", nargs="+", default=["glm-5.2"], choices=list(JUDGES))
    ap.add_argument("--modes", nargs="+", default=[MODE_FULL],
                    choices=[MODE_FULL, MODE_TURN])
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8, help="concurrent dialogues")
    ap.add_argument("--limit", type=int, default=None, help="first N dialogues only")
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--judge-prompt", required=True,
                    help="path to the judge prompt file every judge is run with")
    ap.add_argument("--reproduction-mode", action="store_true",
                    help="require the judge prompt to hash-match the reference prompt")
    args = ap.parse_args()

    paper_prompts.emit_reproduction_notice(args.reproduction_mode)

    # Validate before spending: all judges label with this exact prompt, so a bad file
    # would invalidate the label sets the fine-tunes are compared on.
    judge_prompt = paper_prompts.load(
        "judge",
        path=args.judge_prompt,
        mode=paper_prompts.mode_for(args.reproduction_mode),
    )
    print(f"judge prompt: {judge_prompt.filename} "
          f"({judge_prompt.source}, sha256 {judge_prompt.sha256[:12]})", flush=True)

    dialogues = load_dialogues(args.dialogues)
    if args.limit is not None:
        dialogues = dialogues[: args.limit]
    convs = [d.to_conversation() for d in dialogues]
    ensure_shared_dir(args.outdir)
    print(f"labeling {len(convs)} dialogues from {args.dialogues}", flush=True)

    for jname in args.judges:
        spec = JUDGES[jname]
        model = OpenRouterModel(model=spec["model"], default_opts=spec["opts"])
        for mode in args.modes:
            out = Path(args.outdir) / f"{jname}__{mode}.jsonl"
            total_calls = args.n_samples * (
                len(convs) if mode == MODE_FULL
                else sum(c.num_user_turns for c in convs)
            )
            pmodel = _ProgressModel(model, total_calls, f"{jname}/{mode}")

            def _label(conv, mode=mode, jname=jname, pmodel=pmodel):
                try:
                    labels = label_conversation(
                        pmodel, conv, mode=mode, n_samples=args.n_samples,
                        judge_prompt_path=args.judge_prompt,
                        reproduction_mode=args.reproduction_mode,
                    )
                except Exception as e:  # keep the sweep alive if one dialogue fails
                    print(f"[{jname}/{mode}] {conv.id} FAILED: {e}", flush=True)
                    labels = {}
                return conv.id, labels

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(_label, convs))

            n_turns = 0
            with open_shared(out) as f:
                for conv_id, labels in results:
                    for uti, jl in labels.items():
                        f.write(json.dumps({
                            "dialogue_id": conv_id, "user_turn_index": uti,
                            "judge_label": jl,
                        }) + "\n")
                        n_turns += 1
            out.with_suffix(".provenance.json").write_text(json.dumps({
                "judge": jname,
                "model": spec["model"],
                "opts": spec["opts"],
                "context_mode": mode,
                "n_samples": args.n_samples,
                "dialogues": args.dialogues,
                "judge_prompt": paper_prompts.provenance(judge_prompt),
            }, indent=2))
            print(f"[{jname}/{mode}] wrote {out} ({n_turns} labeled turns)", flush=True)


if __name__ == "__main__":
    main()
