# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Read-only probe: can the 1134 per-turn testset rows be reconstructed into full
conversations for §7.2 full-conversation judging? Prints grouping stats + cost inputs.
Run: ./scripts/eval/run.sh python scripts/eval/inspect_conversation_grouping.py
"""
from collections import defaultdict

from datasets import load_dataset

ds = load_dataset("swordhealth/MindGuard-testset", split="train")


def sig(prompt):
    """Sequence of (role, content) tuples — a hashable conversation prefix key."""
    return tuple((m["role"], m["content"]) for m in prompt)


rows = [dict(prompt=r["prompt"], label=r["label"]) for r in ds]
sigs = [sig(r["prompt"]) for r in rows]
sigset = set(sigs)

# A row is "maximal" if its prompt is not a strict prefix of any other row's prompt.
# Group every row under the maximal conversation(s) it prefixes.
maximal = []
for s in sigset:
    is_prefix_of_other = any(o != s and o[: len(s)] == s for o in sigset)
    if not is_prefix_of_other:
        maximal.append(s)

# For each row, how many maximal conversations does it prefix? (1 = clean nesting)
def prefixes_of(s):
    return [m for m in maximal if m[: len(s)] == s]


multi = sum(1 for s in sigset if len(prefixes_of(s)) > 1)
orphan = sum(1 for s in sigset if len(prefixes_of(s)) == 0)

# Turn coverage: within each maximal conversation, which user-turn depths have a gold row?
depth_by_max = defaultdict(set)
sig_to_rows = defaultdict(list)
for r, s in zip(rows, sigs):
    sig_to_rows[s].append(r)
for s in sigset:
    for m in prefixes_of(s):
        # user-turn index = number of user turns in this prefix
        n_user = sum(1 for role, _ in s if role == "user")
        depth_by_max[m].add(n_user)

covered_all = 0
for m in maximal:
    n_user_total = sum(1 for role, _ in m if role == "user")
    if depth_by_max[m] == set(range(1, n_user_total + 1)):
        covered_all += 1

print(f"total rows:               {len(rows)}")
print(f"distinct prompt sigs:     {len(sigset)}")
print(f"maximal conversations:    {len(maximal)}")
print(f"rows prefixing >1 maximal: {multi}   (0 = clean unique nesting)")
print(f"orphan rows (0 maximal):   {orphan}")
print(f"maximal convs whose EVERY user turn has a gold row: {covered_all}/{len(maximal)}")
print()
ulens = sorted(sum(1 for role, _ in m if role == "user") for m in maximal)
print(f"user-turns per maximal conv: min {ulens[0]}, max {ulens[-1]}, "
      f"median {ulens[len(ulens)//2]}, total gold turns {len(rows)}")
print(f"judge cost inputs: {len(maximal)} convs x N samples x 3 judges")
