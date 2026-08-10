# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Shared primitives for the audit tools (dialogue set, label set): a severity-graded
``Check`` and a table renderer. Kept separate so each audit module only owns its checks,
not the reporting boilerplate.

Severity: ``fail`` = a real defect in the data (drives a non-zero exit), ``warn`` =
worth a human glance, ``ok`` = clean or informational.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Check:
    label: str
    value: str
    severity: str = "ok"  # "ok" | "warn" | "fail"


def sev(bad: int, warn_only: bool = False) -> str:
    """Severity from a defect count: 0 → ok, else fail (or warn if ``warn_only``)."""
    if not bad:
        return "ok"
    return "warn" if warn_only else "fail"


_MARK = {"ok": " ok ", "warn": "WARN", "fail": "FAIL"}


def format_report(report, title: str = "audit") -> str:
    """Render any report exposing ``.checks`` (a list of ``Check``) as an aligned table
    with a verdict line. Reads only ``.checks`` so it works for every audit type."""
    checks = report.checks
    width = max((len(c.label) for c in checks), default=0)
    lines = [title, "=" * len(title)]
    for c in checks:
        lines.append(f"[{_MARK[c.severity]}] {c.label.ljust(width)}  {c.value}")
    n_fail = sum(1 for c in checks if c.severity == "fail")
    n_warn = sum(1 for c in checks if c.severity == "warn")
    verdict = "CLEAN" if n_fail == 0 else f"{n_fail} FAILURE(S)"
    lines.append("-" * len(title))
    lines.append(f"{verdict} — {n_warn} warning(s)")
    return "\n".join(lines)
