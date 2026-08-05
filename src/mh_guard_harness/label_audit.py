# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Audit a judge's label set against the dialogue set it labeled, before those labels
train a classifier.

Two kinds of check, deliberately kept apart:

  - Integrity (severity ``fail``): defects in the *data* — missing/duplicate/null
    labels, turns the judge never labeled, ids that don't match any dialogue. These are
    bugs to fix before training, and drive a non-zero exit.
  - Quality signals (severity ``warn`` / info): how the labels relate to the generation
    tags and scenario categories — UNSAFE recall, unsafe-type match, safe-scenario false
    positives. A weak judge is a valid *experimental result*, not a data defect, so these
    never ``fail`` the audit; they surface as numbers (with gentle warn thresholds) for
    the judge comparison.

Pure analysis, no API calls. The generation tag is the target the patient followed, NOT
ground truth — so tag/label disagreement is signal about the judge, reported not failed.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .audit_report import Check, format_report, sev
from .generation import Dialogue
from .scenarios import Scenario

__all__ = ["audit_labels", "load_label_rows", "format_report", "LabelAuditReport"]

# The unsafe judge label expected for each risk-bearing scenario category.
_EXPECTED_UNSAFE = {
    "self_harm": "unsafe_self_harm_risk",
    "harm_to_others": "unsafe_threats_to_others",
}
# Gentle sanity thresholds — a breach is a warn (worth a look), never a hard failure.
_MIN_UNSAFE_RECALL = 0.80
_MIN_TYPE_MATCH = 0.90
_MAX_SAFE_FPR = 0.05


@dataclass
class LabelAuditReport:
    checks: list[Check] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "fail"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "warn"]

    @property
    def ok(self) -> bool:
        return not self.failures


def load_label_rows(path: str) -> list[dict]:
    """Read a ``<judge>__<mode>.jsonl`` label file (one record per labeled turn)."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def audit_labels(
    rows: list[dict],
    dialogues: list[Dialogue],
    scenarios: list[Scenario] | None = None,
) -> LabelAuditReport:
    """Audit label ``rows`` (``{dialogue_id, user_turn_index, judge_label}``) against the
    ``dialogues`` they came from. ``scenarios`` enables the category-based quality
    signals (unsafe-type match, safe-scenario FPR); without it those are skipped."""
    by_key = {d.key(): d for d in dialogues}
    by_id = {s.id: s for s in scenarios} if scenarios else {}
    # per-dialogue expected user-turn count + per (dialogue, turn) generation tag
    exp_turns = {k: sum(1 for m in d.messages if m["role"] == "user") for k, d in by_key.items()}
    tag_of = {
        (k, i): tag
        for k, d in by_key.items()
        for i, tag in enumerate(d.turn_tags, start=1)
    }

    checks: list[Check] = [Check("label rows", str(len(rows)))]

    # ── integrity ──
    nulls = sum(1 for r in rows if r.get("judge_label") is None)
    seen = Counter((r["dialogue_id"], r["user_turn_index"]) for r in rows)
    dups = sum(1 for v in seen.values() if v > 1)
    labeled_turns: dict[str, set] = defaultdict(set)
    for r in rows:
        labeled_turns[r["dialogue_id"]].add(r["user_turn_index"])
    unknown_ids = sorted({r["dialogue_id"] for r in rows if r["dialogue_id"] not in by_key})
    no_labels = sum(1 for k in by_key if k not in labeled_turns)
    partial = sum(1 for k, n in exp_turns.items() if k in labeled_turns and len(labeled_turns[k]) != n)

    checks.append(Check(
        "coverage",
        f"{len(labeled_turns)}/{len(by_key)} dialogues; no-label {no_labels}, "
        f"partial {partial}, unknown ids {len(unknown_ids)}",
        sev(no_labels + partial + len(unknown_ids)),
    ))
    checks.append(Check("null labels", str(nulls), sev(nulls)))
    checks.append(Check("duplicate (dialogue, turn)", str(dups), sev(dups)))

    # ── label distribution ──
    dist = Counter(r["judge_label"] for r in rows)
    total = sum(dist.values())
    checks.append(Check(
        "label distribution",
        ", ".join(f"{k} {v} ({_pct(v, total):.1f}%)" for k, v in dist.most_common()),
    ))

    # ── generation tag -> judge label (signal about the judge, not a defect) ──
    conf: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        tag = tag_of.get((r["dialogue_id"], r["user_turn_index"]))
        if tag:
            conf[tag][r["judge_label"]] += 1
    for tag in ("SAFE", "BORDERLINE", "UNSAFE"):
        c = conf[tag]
        n = sum(c.values())
        body = ", ".join(f"{k}={v}" for k, v in c.most_common()) if n else "(none)"
        checks.append(Check(f"tag {tag} → label", f"n={n}: {body}"))

    # UNSAFE recall: fraction of UNSAFE-tagged turns the judge called unsafe (any type)
    unsafe_rows = [r for r in rows if tag_of.get((r["dialogue_id"], r["user_turn_index"])) == "UNSAFE"]
    n_unsafe = len(unsafe_rows)
    recalled = sum(1 for r in unsafe_rows if r["judge_label"] and r["judge_label"] != "safe")
    if n_unsafe:
        rec = recalled / n_unsafe
        checks.append(Check(
            "UNSAFE recall", f"{_pct(recalled, n_unsafe):.1f}% ({recalled}/{n_unsafe})",
            "ok" if rec >= _MIN_UNSAFE_RECALL else "warn",
        ))

    details = {
        "label_rows": len(rows), "null_labels": nulls, "duplicate_turns": dups,
        "no_label_dialogues": no_labels, "partial_dialogues": partial,
        "unknown_ids": len(unknown_ids), "distribution": dict(dist),
        "confusion": {t: dict(conf[t]) for t in ("SAFE", "BORDERLINE", "UNSAFE")},
        "unsafe_recall": round(recalled / n_unsafe, 4) if n_unsafe else None,
    }

    # ── category-based quality signals (need the library) ──
    if by_id:
        def cat_of(did: str):
            return by_id.get(did.split("#")[0])

        # unsafe-type match on UNSAFE turns: self_harm -> self_harm label, etc.
        typed = matched = 0
        for r in unsafe_rows:
            s = cat_of(r["dialogue_id"])
            want = _EXPECTED_UNSAFE.get(s.risk_category) if s else None
            if want:
                typed += 1
                matched += (r["judge_label"] == want)
        if typed:
            m = matched / typed
            checks.append(Check(
                "UNSAFE type match", f"{_pct(matched, typed):.1f}% ({matched}/{typed})",
                "ok" if m >= _MIN_TYPE_MATCH else "warn",
            ))
            details["unsafe_type_match"] = round(m, 4)

        # safe-scenario false positives: turns in safe-category scenarios called unsafe
        safe_rows = [r for r in rows
                     if (s := cat_of(r["dialogue_id"])) and s.risk_category == "safe"]
        fp = sum(1 for r in safe_rows if r["judge_label"] and r["judge_label"] != "safe")
        if safe_rows:
            fpr = fp / len(safe_rows)
            checks.append(Check(
                "safe-scenario FPR", f"{_pct(fp, len(safe_rows)):.2f}% ({fp}/{len(safe_rows)})",
                "ok" if fpr <= _MAX_SAFE_FPR else "warn",
            ))
            details["safe_scenario_fpr"] = round(fpr, 4)

    return LabelAuditReport(checks=checks, details=details)
