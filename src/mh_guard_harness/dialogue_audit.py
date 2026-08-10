# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Structural + quality audit of a generated dialogue set, so a run can be
sanity-checked before spending on judge labeling.

The runner's own signals (``*.failures.json`` / ``*.flagged.json``) catch calls that
errored or read as refusals. This goes a layer deeper on the *content that succeeded*:
coverage against the scenario library, structural integrity (empty/misaligned/
non-alternating turns), leftover formatting the normalizer should have removed, refusal
text the generation heuristic would have flagged, and degeneracy (repeated or templated
turns) — the failure modes that pass silently because nothing threw.

Pure analysis, no API calls. ``audit_dialogues`` returns a structured report;
``format_report`` renders the table. Severity: ``fail`` = a real defect in the data,
``warn`` = worth a human glance, ``ok`` = clean or informational.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

from .audit_report import Check, format_report, sev
from .generation import Dialogue, looks_like_refusal
from .scenarios import Scenario

__all__ = ["audit_dialogues", "format_report", "AuditReport"]

# A leftover leading role label the normalizer is meant to strip ("Patient:", "You -").
_ROLE_PREFIX = re.compile(r"^(patient|client|you|me|user)\s*[:\-]", re.IGNORECASE)
_VERY_SHORT = 3  # chars; a "turn" this short is almost certainly degenerate


@dataclass
class AuditReport:
    n_dialogues: int
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


def audit_dialogues(
    dialogues: list[Dialogue], scenarios: list[Scenario] | None = None
) -> AuditReport:
    """Audit a dialogue set. ``scenarios`` (the library the set was generated from)
    enables coverage, turn-count, category, and UNSAFE-turn checks; without it those are
    skipped and the rest still run."""
    by_id = {s.id: s for s in scenarios} if scenarios else {}
    n = len(dialogues)
    checks: list[Check] = [Check("dialogues", str(n))]

    patient_turns: list[str] = []
    clinician_turns: list[str] = []
    openings: list[str] = []
    empty_p = empty_c = role_leak = multiline = very_short = refusals = 0
    intra_rep = bad_alt = tag_mismatch = flagged = 0
    unsafe_lens: list[int] = []

    for d in dialogues:
        users = [m["content"] for m in d.messages if m["role"] == "user"]
        clins = [m["content"] for m in d.messages if m["role"] == "assistant"]
        patient_turns.extend(users)
        clinician_turns.extend(clins)
        if users:
            openings.append(users[0])
        if len(users) != len(set(users)):
            intra_rep += 1
        if d.flags:
            flagged += 1
        roles = [m["role"] for m in d.messages]
        if roles != ["user" if i % 2 == 0 else "assistant" for i in range(len(roles))]:
            bad_alt += 1
        sc = by_id.get(d.scenario_id)
        if sc and len(users) != len(sc.turn_script):
            tag_mismatch += 1
        for i, t in enumerate(users):
            if not t.strip():
                empty_p += 1
            if "\n" in t:
                multiline += 1
            if _ROLE_PREFIX.match(t):
                role_leak += 1
            if len(t.strip()) < _VERY_SHORT:
                very_short += 1
            if looks_like_refusal(t):
                refusals += 1
            if sc and i < len(sc.turn_script) and sc.turn_script[i] == "UNSAFE":
                unsafe_lens.append(len(t))
        for c in clins:
            if not c.strip():
                empty_c += 1

    # ── coverage (needs the library) ──
    if by_id:
        seen = {d.scenario_id for d in dialogues}
        missing = [sid for sid in by_id if sid not in seen]
        unknown = sorted({d.scenario_id for d in dialogues if d.scenario_id not in by_id})
        dup_keys = [k for k, v in Counter(d.key() for d in dialogues).items() if v > 1]
        checks.append(Check(
            "coverage",
            f"{len(seen)}/{len(by_id)} scenarios; missing {len(missing)}, "
            f"unknown {len(unknown)}, dup keys {len(dup_keys)}",
            sev(len(missing) + len(unknown) + len(dup_keys)),
        ))
    else:
        checks.append(Check("coverage", "skipped (no library given)"))

    # ── structural integrity ──
    checks.append(Check("empty patient turns", str(empty_p), sev(empty_p)))
    if by_id:
        checks.append(Check("turn count == script", f"{tag_mismatch} mismatched", sev(tag_mismatch)))
    checks.append(Check("alternation", f"{bad_alt} malformed", sev(bad_alt)))
    checks.append(Check("clinician empty turns", str(empty_c), sev(empty_c)))
    checks.append(Check("intra-dialogue repeats", str(intra_rep), sev(intra_rep)))

    # ── output hygiene / review signals ──
    checks.append(Check("dialogues flagged (refusal)", str(flagged), sev(flagged, warn_only=True)))
    checks.append(Check("refusal-looking turns", str(refusals), sev(refusals, warn_only=True)))
    checks.append(Check("role-label leak", str(role_leak), sev(role_leak)))
    checks.append(Check("multiline turns", str(multiline), sev(multiline, warn_only=True)))
    checks.append(Check("very short turns (<3)", str(very_short), sev(very_short, warn_only=True)))

    # ── degeneracy ──
    dup_open = sum(v - 1 for v in Counter(openings).values() if v > 1)
    checks.append(Check("shared openings", str(dup_open), sev(dup_open, warn_only=True)))
    p_uniq = 100 * len(set(patient_turns)) / len(patient_turns) if patient_turns else 100.0
    checks.append(Check(
        "patient turn uniqueness",
        f"{p_uniq:.1f}% ({len(set(patient_turns))}/{len(patient_turns)})",
        "ok" if p_uniq >= 99.9 else "warn",
    ))
    c_uniq = 100 * len(set(clinician_turns)) / len(clinician_turns) if clinician_turns else 100.0
    checks.append(Check(
        "clinician turn uniqueness",
        f"{c_uniq:.1f}% ({len(set(clinician_turns))}/{len(clinician_turns)})",
        "ok" if c_uniq >= 99.9 else "warn",
    ))

    # ── informational (needs the library) ──
    if by_id:
        cat = Counter(by_id[d.scenario_id].risk_category for d in dialogues if d.scenario_id in by_id)
        checks.append(Check("category distribution",
                            ", ".join(f"{k} {v}" for k, v in sorted(cat.items()))))
        if unsafe_lens:
            checks.append(Check(
                "UNSAFE turns",
                f"{len(unsafe_lens)}; chars min {min(unsafe_lens)} / "
                f"median {int(statistics.median(unsafe_lens))} / max {max(unsafe_lens)}",
            ))

    details = {
        "n_dialogues": n,
        "patient_turns": len(patient_turns),
        "clinician_turns": len(clinician_turns),
        "empty_patient_turns": empty_p,
        "empty_clinician_turns": empty_c,
        "turn_count_mismatch": tag_mismatch,
        "bad_alternation": bad_alt,
        "dialogues_flagged": flagged,
        "refusal_turns": refusals,
        "role_label_leak": role_leak,
        "multiline_turns": multiline,
        "very_short_turns": very_short,
        "intra_dialogue_repeats": intra_rep,
        "shared_openings": dup_open,
        "patient_uniqueness_pct": round(p_uniq, 2),
        "clinician_uniqueness_pct": round(c_uniq, 2),
        "unsafe_turns": len(unsafe_lens),
    }
    return AuditReport(n_dialogues=n, checks=checks, details=details)
