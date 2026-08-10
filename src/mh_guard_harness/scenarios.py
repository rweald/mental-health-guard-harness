# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Scenario schema — the generation unit for the training lineage. A Scenario is a plan
for one synthetic patient↔clinician dialogue: who the patient is (``Persona``), what risk
it carries, and a per-turn risk trajectory (``turn_script``) the patient generator follows.

Crucially, ``turn_script`` tags are the *generation target*, not the training labels:
they steer how the patient escalates/de-escalates, but the judge still labels every
user turn independently (that's the whole method). So nothing here ever becomes a
label — it only shapes the dialogue.

This is training-lineage code. It never imports or touches the evaluation lineage
(the licensed evaluation set); the two only meet at the shared, read-only judge in
``judging.py``.

Adapting the vocabularies
-------------------------
The four controlled vocabularies below are meant to be edited — they encode one
population and one taxonomy, and a different study will want different ones. Unlike
``Persona``, whose field list is fully derived from the dataclass, **these are only
partly derived.** Each has companion sites that must be edited with it, and every
failure mode is silent: the scenario validates, the run completes, and the thing you
added simply never appears (or worse, is generated as though it were safe).

Co-edit each vocabulary with:

- ``RISK_CATEGORIES`` → ``authoring.DEFAULT_CATEGORY_TARGET`` (else no cells are ever
  allocated to it) · ``authoring.DEFAULT_SUBCAT_WEIGHTS`` · ``authoring._taxonomy_block``
  (else the authoring model is never told what the category means) ·
  ``generation._tag_guidance`` (else an UNSAFE turn falls through to the SAFE branch and
  is generated as ordinary talk) · ``TARGET_BALANCE`` below. A new category also needs a
  judge label to carry it — see the note on label spaces at the end.
- ``SUBCATEGORIES`` → ``authoring.DEFAULT_SUBCAT_WEIGHTS[category]``. A subcategory
  present here but absent there validates fine and is never authored.
- ``TRAJECTORIES`` → ``authoring.DEFAULT_TRAJ_WEIGHTS`` and
  ``authoring._turn_script_block``, which tells the model what shape each trajectory's
  ``turn_script`` should take.
- ``RISK_TAGS`` → ``generation._tag_guidance`` (per-tag instruction to the patient
  model) and ``authoring._turn_script_block``. ``errors()`` also compares tags to the
  literal ``"UNSAFE"`` / ``"BORDERLINE"``.

Two strings are load-bearing beyond their vocabulary: ``errors()`` special-cases the
category ``"safe"``, and ``generation._tag_guidance`` branches on ``"self_harm"`` and
``"harm_to_others"`` by name. Renaming those means updating those branches.

**Scenario risk categories and judge labels are different vocabularies.** These describe
what a dialogue was *authored to contain*; the judge emits its own labels
(``judging.LABELS``) which become training targets via
``training.dataset._LABEL_TO_TARGET``. They correspond today but are not linked in code,
so a new risk category needs the judge prompt to emit a matching label and that mapping
extended — the mapping at least fails loudly on an unknown label rather than silently.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields

# ── controlled vocabularies ──────────────────────────────────────────────────
# Editable — see "Adapting the vocabularies" in the module docstring for the companion
# sites each one must be edited alongside. Adding to any of these alone is a silent no-op.

# Overall risk *type* of a scenario. A "safe" scenario carries no unsafe turns; the
# two unsafe types mirror the judge taxonomy (self-harm vs harm-to-others).
# Co-edit: authoring.DEFAULT_CATEGORY_TARGET, DEFAULT_SUBCAT_WEIGHTS, _taxonomy_block;
# generation._tag_guidance; TARGET_BALANCE below. "safe" is special-cased in errors().
RISK_CATEGORIES = ("safe", "self_harm", "harm_to_others")

# Subcategory within each risk type. These are authoring buckets, not labels —
# over-represent the borderline ones (passive ideation, metaphor, sustained ambiguity),
# which is where judges are most likely to disagree.
# Co-edit: authoring.DEFAULT_SUBCAT_WEIGHTS[category] — a subcategory listed here but
# missing there validates fine and is never assigned a cell to author.
SUBCATEGORIES: dict[str, tuple[str, ...]] = {
    "safe": (
        "general_stress",
        "relationship",
        "work",
        "venting",
        "grief",
        "loneliness",
        "health_anxiety",
    ),
    "self_harm": (
        "passive_ideation",
        "direct_ideation",
        "method_inquiry",
        "metaphor",
        "past_attempt",
        "hopelessness",
    ),
    "harm_to_others": (
        "violent_ideation",
        "direct_threat",
        "means_planning",
        "target_identification",
        "abuse_neglect",
    ),
}

# Shape of the risk trajectory across the dialogue (descriptive metadata; the concrete
# per-turn plan lives in ``turn_script``).
# Co-edit: authoring.DEFAULT_TRAJ_WEIGHTS (else no cells) and authoring._turn_script_block
# (which tells the model what shape each trajectory's turn_script should take).
TRAJECTORIES = ("escalation", "sustained_ambiguity", "de_escalation")

# Per-turn risk tag the patient generator aims for on that turn. SAFE = ordinary talk;
# BORDERLINE = ambiguous / passive / metaphorical; UNSAFE = clear intent of the
# scenario's risk_category. Ordered by severity for trajectory checks.
# Co-edit: generation._tag_guidance (the per-tag instruction to the patient model) and
# authoring._turn_script_block. errors() also tests "UNSAFE"/"BORDERLINE" by name.
RISK_TAGS = ("SAFE", "BORDERLINE", "UNSAFE")
_TAG_SEVERITY = {tag: i for i, tag in enumerate(RISK_TAGS)}

# Reference risk balance, a target for the scenario library's category mix. Descriptive
# only — the quota actually authored comes from authoring.DEFAULT_CATEGORY_TARGET, so
# change both together or they will disagree.
TARGET_BALANCE = {"safe": 0.62, "self_harm": 0.21, "harm_to_others": 0.16}


# ── schema ───────────────────────────────────────────────────────────────────
@dataclass
class Persona:
    """The patient attribute set. Fields default to empty so partial personas are usable
    while authoring, but a library-quality persona fills the full set — completeness is a
    quality lever for generation.

    **This field list is the schema, and editing it is the supported way to change it.**
    The set below follows the reference method's archetype attributes, which suit general
    adult mental-health support; a different population probably wants different fields
    (adolescents: school, peer group; perinatal: pregnancy stage, birth history). Add,
    remove and rename freely — everything downstream is derived, and nothing anywhere in
    the codebase reaches for a persona field by name.

    Editing this list automatically updates:

    - the attribute list the authoring model is given (``authoring._persona_block``)
    - the persona block rendered into the patient prompt (``generation._persona_block``)
    - JSON serialization in both directions (``to_dict`` / ``from_dict``)

    Four things to know before you edit:

    1. **Every field needs a default.** ``Persona()`` is constructed with no arguments to
       enumerate the schema, so a field without one breaks prompt building.
    2. **``backstory`` is deliberately excluded** from both attribute listings — it is
       narrative prose, not an attribute, and belongs in its own part of the prompt. Give
       any other free-text narrative field the same treatment.
    3. **Existing libraries keep loading.** ``from_dict`` drops keys it doesn't recognise
       and defaults ones that are absent, so a library authored under the old field list
       still parses. It will simply be missing your new attributes.
    4. **Don't mix libraries across schema changes in one comparison.** Changing the
       fields changes the authoring prompt, so scenarios authored before and after are not
       strictly comparable. Re-author rather than blend.
    """

    name: str = ""
    sex: str = ""
    gender_identity: str = ""
    sexual_orientation: str = ""
    age: int | None = None
    race: str = ""
    thought_process: str = ""
    outlook: str = ""
    conversation_style: str = ""
    recent_mood: str = ""
    education: str = ""
    profession: str = ""
    employment: str = ""
    finances: str = ""
    siblings: str = ""
    relationship_status: str = ""
    living_situation: str = ""
    exercise: str = ""
    sleep: str = ""
    mindfulness_attitude: str = ""
    region: str = ""
    depressive_symptoms: str = ""
    anxious_symptoms: str = ""
    # Optional 4-paragraph second-person backstory (formative → onset → current
    # functioning), scaled to severity. Left None when not authored.
    backstory: str | None = None


@dataclass
class Scenario:
    """One dialogue plan. ``turn_script`` has one risk tag per *patient* turn, so its
    length is the number of user turns to generate (~6–10).

    Adding a field, unlike adding one to ``Persona``, takes two steps:

    1. **Give it a default.** ``from_dict`` drops keys it doesn't recognise but cannot
       invent required ones, so a field without a default makes every library authored
       before the change fail to load with a ``TypeError``. (The seven fields above
       without defaults are required for exactly that reason — a scenario missing one is
       not a scenario.) This is the one failure in this module that is loud rather than
       silent; take the hint and prefer a default.
    2. **Ask for it in the authoring prompt.** The authoring prompt does not derive from
       this dataclass the way the persona attribute list does — it has a hand-written
       block per field plus a JSON template, both in ``authoring.build_author_prompt``.
       A field added here but not there is simply never authored, and every scenario
       gets the default forever.

    ``arc`` and ``notes`` are the model for an optional field: defaulted, tolerated when
    absent, and ``notes`` is deliberately never shown to any model."""

    id: str
    risk_category: str  # one of RISK_CATEGORIES
    subcategory: str  # one of SUBCATEGORIES[risk_category]
    persona: Persona
    trajectory: str  # one of TRAJECTORIES
    turn_script: list[str]  # per-patient-turn tag from RISK_TAGS
    opening_instruction: str  # how the patient opens turn 1
    arc: str = ""  # optional "what's happening" narrative (paper Fig 5/6), shown to the PLM
    notes: str = ""  # authoring / clinical-provenance notes (never shown to a model)

    @property
    def max_turns(self) -> int:
        return len(self.turn_script)


# ── validation ───────────────────────────────────────────────────────────────
def errors(scenario: Scenario) -> list[str]:
    """Return a list of structural problems with ``scenario`` (empty = valid). Catches
    authoring mistakes before an expensive generation run spends on a broken plan.

    This is the single authority on what a scenario must satisfy — nothing else
    re-validates, so a rule that is not here is not enforced. Membership checks read the
    vocabularies at call time and so pick up edits to them automatically; the semantic
    rules below do not. If you add a risk category with its own invariant (say, one that
    must never de-escalate), that rule has to be written here by hand."""
    errs: list[str] = []
    s = scenario

    if not s.id:
        errs.append("id is empty")
    if s.risk_category not in RISK_CATEGORIES:
        errs.append(f"risk_category {s.risk_category!r} not in {RISK_CATEGORIES}")
        return errs  # later checks key off risk_category; bail to avoid a KeyError
    if s.subcategory not in SUBCATEGORIES[s.risk_category]:
        errs.append(
            f"subcategory {s.subcategory!r} not valid for {s.risk_category!r} "
            f"(expected one of {SUBCATEGORIES[s.risk_category]})"
        )
    if s.trajectory not in TRAJECTORIES:
        errs.append(f"trajectory {s.trajectory!r} not in {TRAJECTORIES}")

    if not s.turn_script:
        errs.append("turn_script is empty (need >=1 patient turn)")
    else:
        bad = [t for t in s.turn_script if t not in RISK_TAGS]
        if bad:
            errs.append(f"turn_script has invalid tags {bad} (allowed {RISK_TAGS})")

    # A safe scenario may touch BORDERLINE but must never reach UNSAFE. The authoring
    # prompt still targets the paper's all-SAFE ideal for safe cells; this leniency is a
    # spend optimization — when the model incidentally scripts a BORDERLINE (never UNSAFE)
    # turn in a safe scenario, we keep that paid output instead of rejecting and paying to
    # regenerate it (and it doubles as a useful ambiguous-but-safe hard negative). An
    # UNSAFE turn in a safe scenario is a real mislabel and is still rejected. An unsafe
    # scenario must actually reach some risk (else it is mislabeled as safe and pollutes
    # the risk-category balance).
    tags = [t for t in s.turn_script if t in RISK_TAGS]
    if s.risk_category == "safe":
        if any(t == "UNSAFE" for t in tags):
            errs.append("safe scenario has UNSAFE turns in turn_script")
    elif tags and not any(t in ("BORDERLINE", "UNSAFE") for t in tags):
        errs.append(
            f"{s.risk_category!r} scenario never leaves SAFE — no BORDERLINE/UNSAFE turn"
        )

    if not s.opening_instruction:
        errs.append("opening_instruction is empty")
    return errs


def validate(scenario: Scenario) -> None:
    """Raise ``ValueError`` if ``scenario`` is structurally invalid."""
    errs = errors(scenario)
    if errs:
        raise ValueError(f"invalid scenario {scenario.id!r}: " + "; ".join(errs))


def validate_library(scenarios: list[Scenario]) -> dict:
    """Validate a whole library: raise on any invalid scenario or duplicate id, and
    return a summary (count, per-risk-category balance, drift from TARGET_BALANCE)."""
    seen: set[str] = set()
    for s in scenarios:
        validate(s)
        if s.id in seen:
            raise ValueError(f"duplicate scenario id {s.id!r}")
        seen.add(s.id)

    n = len(scenarios)
    counts = Counter(s.risk_category for s in scenarios)
    balance = {c: counts.get(c, 0) / n for c in RISK_CATEGORIES} if n else {}
    drift = {c: round(balance.get(c, 0) - TARGET_BALANCE[c], 3) for c in TARGET_BALANCE}
    return {"n": n, "counts": dict(counts), "balance": balance, "drift": drift}


# ── (de)serialization ────────────────────────────────────────────────────────
def to_dict(scenario: Scenario) -> dict:
    return asdict(scenario)


def _known(cls, data: dict) -> dict:
    """Keep only keys that are fields of ``cls``. A model authoring scenarios will add
    fields the schema doesn't have (e.g. an extra ``additional_notes`` on a persona);
    dropping unknowns lets the object build instead of crashing the whole draft."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def from_dict(d: dict) -> Scenario:
    d = dict(d)
    persona = d.pop("persona", {})
    return Scenario(persona=Persona(**_known(Persona, persona)), **_known(Scenario, d))


def load_library(path: str) -> list[Scenario]:
    """Load a scenario library from a JSON file (a list of scenario dicts)."""
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text())
    return [from_dict(d) for d in payload]


def dump_library(scenarios: list[Scenario], path: str) -> None:
    """Write a scenario library to a JSON file (world-writable, matching io_utils)."""
    import json

    from .io_utils import open_shared

    with open_shared(path) as f:
        f.write(json.dumps([to_dict(s) for s in scenarios], indent=2))
