# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Two-agent generation loop: a patient model (PLM) and a clinician
model (CLM) take turns to produce one synthetic dialogue from a Scenario. The patient
follows the scenario's per-turn risk trajectory; the clinician is scenario-blind and
its turns are context only (never labeled).

The output ``Dialogue`` carries the messages user-first, plus the generation-target
tags as provenance. ``to_conversation`` turns it into the exact ``Conversation`` shape
``judging.label_conversation`` consumes — so both arms label the identical dialogues
with the identical judge code. The tags are dropped in that conversion: the judge only
ever sees user-turn text, never the plan that produced it.

Training-lineage code. It borrows the neutral ``Conversation``/``GoldTurn`` containers
(the judge's input type, shared by design) but no testset *data* crosses here.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .conversations import Conversation, GoldTurn
from .models import Model
from .scenarios import Persona, Scenario, validate

# ── prompts ──────────────────────────────────────────────────────────────────
# Stable role framing for the patient; the per-turn specifics go in the user message.
PATIENT_SYSTEM = (
    "You are role-playing a person (the PATIENT) talking to a mental-health chat "
    "assistant. This produces synthetic dialogue for training a turn-level safety "
    "classifier. Speak only as the patient, in the first person. Output ONLY your next "
    "message — no narration, stage directions, quotation marks, or role labels. Keep it "
    "to a natural chat length (1–4 sentences)."
)

# Fixed, scenario-blind clinician prompt: it never sees the persona or risk trajectory.
CLINICIAN_SYSTEM = (
    "You are a warm, professional mental-health clinician in a text conversation with a "
    "client. Respond supportively to the client's latest message: reflect and validate "
    "what they said, and ask a gentle, open question to keep them talking. Keep replies "
    "brief (2–4 sentences). Reply with only your message, in the first person."
)


def _tag_guidance(risk_category: str, subcategory: str, tag: str) -> str:
    """Per-turn instruction to the patient generator for a given risk tag. Steers the
    trajectory without authoring operational detail — the model supplies the words."""
    if tag == "SAFE":
        return (
            "Talk about ordinary life, stress, or feelings. Do NOT express any intent "
            "to harm yourself or anyone else."
        )
    if risk_category == "self_harm":
        if tag == "BORDERLINE":
            return (
                "Convey hopelessness or dark feelings in an ambiguous, indirect way "
                "(passive wishes, metaphor, being tired of it all) — with NO explicit "
                "statement of intent or a plan to harm yourself."
            )
        return (
            f"Clearly convey present self-harm risk consistent with '{subcategory}' "
            "(e.g. stated intent, planning, or seeking means). Stay in the patient's "
            "voice; do not describe graphic methods."
        )
    if risk_category == "harm_to_others":
        if tag == "BORDERLINE":
            return (
                "Express anger or frustration toward a specific person as venting only "
                "— with NO concrete threat, plan, target, or means."
            )
        return (
            f"Clearly convey intent or planning to harm an identifiable other, "
            f"consistent with '{subcategory}' (e.g. a threat, a named target, or "
            "means). Stay in the patient's voice; do not provide operational detail."
        )
    return "Talk about ordinary life, stress, or feelings."


def _persona_block(p: Persona) -> str:
    fields = {
        k: v for k, v in asdict(p).items()
        if k != "backstory" and v not in ("", None)
    }
    if not fields:
        return "- (unspecified)"
    return "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in fields.items())


def _render_history(messages: list[dict]) -> str:
    who = {"user": "You", "assistant": "Therapist"}
    return "\n".join(f"{who[m['role']]}: {m['content']}" for m in messages)


def build_patient_messages(
    scenario: Scenario, turn_index: int, history: list[dict]
) -> list[dict]:
    """Messages for the PLM to produce patient turn ``turn_index`` (0-based). ``history``
    is the dialogue so far (empty on the first turn, else ending on a clinician reply)."""
    p = scenario.persona
    guidance = _tag_guidance(
        scenario.risk_category, scenario.subcategory, scenario.turn_script[turn_index]
    )
    parts = [f"PATIENT PROFILE:\n{_persona_block(p)}"]
    if p.backstory:
        parts.append(f"BACKSTORY:\n{p.backstory}")
    if scenario.arc:
        parts.append(f"SCENARIO ARC:\n{scenario.arc}")
    if history:
        parts.append(f"CONVERSATION SO FAR:\n{_render_history(history)}")
        parts.append(
            f"YOUR NEXT MESSAGE (turn {turn_index + 1} of {scenario.max_turns}):\n"
            f"{guidance}"
        )
    else:
        parts.append(f"HOW TO OPEN: {scenario.opening_instruction}")
        parts.append(f"YOUR FIRST MESSAGE:\n{guidance}")
    return [
        {"role": "system", "content": PATIENT_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_clinician_messages(messages: list[dict]) -> list[dict]:
    """Messages for the CLM to reply to the latest patient turn. Real roles: patient =
    user, clinician = assistant. ``messages`` must end on a patient (user) turn."""
    return [{"role": "system", "content": CLINICIAN_SYSTEM}, *messages]


# ── output hygiene ───────────────────────────────────────────────────────────
# A leading role label the patient model sometimes emits despite being told not to
# ("Patient:", "You -", "Me:"). Requires an explicit ":"/"-" delimiter so ordinary
# sentences ("You know, I…") are never touched.
_ROLE_PREFIX = re.compile(r"^(patient|client|you|me|user)\s*[:\-]\s*", re.IGNORECASE)

# A message that begins like a provider refusal. Conservative and anchored at the start
# (refusals lead) — this only *flags* a turn for review, never drops it, so a rare false
# positive costs a review item, not data.
_REFUSAL_RE = re.compile(
    r"^\s*(i'?m sorry[,.]? (but|i)|i can'?t (help|continue|assist|do that|engage|role[- ]?play)|"
    r"i cannot (help|continue|assist|do that|engage|comply|role[- ]?play)|"
    r"i (won'?t|will not) (be able|continue|help)|i'?m (not able|unable) to|"
    r"i am unable to|i must decline|as an ai|i'?m an ai\b)",
    re.IGNORECASE,
)


def _normalize_turn(text: str) -> str:
    """Clean one model turn for storage and the numbered judge transcript: unwrap a
    whole-message quote, strip a leading role label, and collapse internal whitespace so
    a turn is exactly one line (``judging.render_transcript`` puts one turn per line)."""
    t = (text or "").strip()
    for q in ('"', "'"):
        # count == 2 means the quotes truly wrap the message (not stray inner quotes)
        if len(t) >= 2 and t[0] == q and t[-1] == q and t.count(q) == 2:
            t = t[1:-1].strip()
            break
    t = _ROLE_PREFIX.sub("", t, count=1)
    return re.sub(r"\s+", " ", t).strip()


def looks_like_refusal(text: str) -> bool:
    """Heuristic: does this patient turn read as a provider refusal rather than in-role
    patient text? Used to flag (not drop) dialogues for human review."""
    return bool(_REFUSAL_RE.match(text or ""))


def _complete_turn(
    model: Model, messages: list[dict], opts: dict, *, who: str, scenario: Scenario, turn: int
) -> str:
    """One turn call, normalized and guarded against empty output. Retries once (with a
    perturbed seed so a seed-deterministic empty can differ); if still empty, raises so
    the runner isolates and *counts* the dialogue rather than silently embedding a blank
    turn that would then be labeled and pollute training."""
    for attempt in range(2):
        call_opts = opts if attempt == 0 else {**opts, "seed": (opts.get("seed") or 0) + 7919}
        text = _normalize_turn(model.complete(messages, **call_opts).text)
        if text:
            return text
    raise ValueError(
        f"{who} model returned empty output for scenario {scenario.id!r} "
        f"turn {turn + 1} (after one retry)"
    )


# ── dialogue ─────────────────────────────────────────────────────────────────
@dataclass
class Dialogue:
    """One generated conversation. ``messages`` is user-first and alternating; each user
    turn has a matching ``turn_tags`` entry (the generation target, provenance only).
    ``flags`` records non-fatal quality concerns (e.g. a turn that reads as a refusal)
    for human review — provenance only, never read by the judge."""

    scenario_id: str
    seed: int
    messages: list[dict] = field(default_factory=list)
    turn_tags: list[str] = field(default_factory=list)
    flags: list[dict] = field(default_factory=list)

    def key(self) -> str:
        """Stable id for this dialogue: scenario + seed. Both arms label this same key."""
        return f"{self.scenario_id}#seed{self.seed}"

    def user_turns(self) -> list[str]:
        return [m["content"] for m in self.messages if m["role"] == "user"]

    def to_conversation(self) -> Conversation:
        """Adapt to the judge's input type, marking *every* user turn as a target (the
        judge assigns the label — the placeholder ``label=""`` is never read)."""
        gold_turns, uti = [], 0
        for i, m in enumerate(self.messages):
            if m["role"] == "user":
                uti += 1
                gold_turns.append(GoldTurn(user_turn_index=uti, msg_index=i, label=""))
        return Conversation(id=self.key(), messages=self.messages, gold_turns=gold_turns)


def generate_dialogue(
    scenario: Scenario,
    plm: Model,
    clm: Model,
    *,
    seed: int = 0,
    plm_opts: dict | None = None,
    clm_opts: dict | None = None,
    final_clinician_reply: bool = False,
) -> Dialogue:
    """Run the patient↔clinician loop for one scenario and seed.

    One patient turn per ``turn_script`` entry; a clinician reply follows each patient
    turn except (by default) the last — nothing would react to it and the judge never
    reads clinician turns, so we skip that paid call. ``seed`` is threaded to providers
    that honor it and keys the stored dialogue so both arms label the identical set.
    """
    validate(scenario)
    plm_opts = {"seed": seed, **(plm_opts or {})}
    clm_opts = {"seed": seed, **(clm_opts or {})}

    messages: list[dict] = []
    flags: list[dict] = []
    n = scenario.max_turns
    for i in range(n):
        patient = _complete_turn(
            plm, build_patient_messages(scenario, i, messages), plm_opts,
            who="patient", scenario=scenario, turn=i,
        )
        if looks_like_refusal(patient):
            flags.append({"turn": i + 1, "reason": "possible_refusal", "text": patient[:200]})
        messages.append({"role": "user", "content": patient})
        if i < n - 1 or final_clinician_reply:
            reply = _complete_turn(
                clm, build_clinician_messages(messages), clm_opts,
                who="clinician", scenario=scenario, turn=i,
            )
            messages.append({"role": "assistant", "content": reply})

    return Dialogue(
        scenario_id=scenario.id,
        seed=seed,
        messages=messages,
        turn_tags=list(scenario.turn_script),
        flags=flags,
    )


# ── (de)serialization ────────────────────────────────────────────────────────
def dialogue_to_dict(d: Dialogue) -> dict:
    return asdict(d)


def _known(cls, data: dict) -> dict:
    """Keep only keys that are fields of ``cls`` — so a stored dialogue carrying an extra
    field (a later schema, a hand-added annotation) loads instead of crashing the whole
    labeling run on one line. Mirrors ``scenarios._known``."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def dialogue_from_dict(d: dict) -> Dialogue:
    return Dialogue(**_known(Dialogue, d))


def load_dialogues(path: str) -> list[Dialogue]:
    """Read a dialogues JSONL file (one dialogue per line) into Dialogue objects."""
    out: list[Dialogue] = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(dialogue_from_dict(json.loads(line)))
    return out


def dump_dialogues(dialogues: list[Dialogue], path: str) -> None:
    """Write dialogues as JSONL (world-writable, matching io_utils)."""
    from .io_utils import open_shared

    with open_shared(path) as f:
        for d in dialogues:
            f.write(json.dumps(dialogue_to_dict(d)) + "\n")
