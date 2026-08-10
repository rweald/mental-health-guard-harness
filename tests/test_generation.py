# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Two-agent generation loop: structure, prompts, seed threading, and the adapter that
feeds a generated dialogue into the shared judge (label_conversation)."""

from mh_guard_harness.generation import (
    _normalize_turn,
    build_clinician_messages,
    dialogue_from_dict,
    dialogue_to_dict,
    dump_dialogues,
    generate_dialogue,
    load_dialogues,
    looks_like_refusal,
)
from mh_guard_harness.judging import MODE_FULL, label_conversation
from mh_guard_harness.models import Completion
from mh_guard_harness.scenarios import Persona, Scenario


class FakeModel:
    """Records every call and returns a distinct, whitespace-padded reply per call so
    tests can trace ordering and confirm the loop strips text."""

    def __init__(self, tag):
        self.tag = tag
        self.calls = []

    def complete(self, messages, **opts):
        self.calls.append({"messages": messages, "opts": opts})
        return Completion(text=f"  {self.tag}-{len(self.calls)}  ", raw=None)


class ConstModel:
    """Returns a fixed reply on every call (for refusal / empty-output cases)."""

    def __init__(self, text):
        self._text = text

    def complete(self, messages, **opts):
        return Completion(text=self._text, raw=None)


class SeqModel:
    """Returns queued replies in order (last repeats), to exercise empty-then-retry."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def complete(self, messages, **opts):
        r = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return Completion(text=r, raw=None)


class FakeJudge:
    """Judge stub that labels every turn 'safe' (enough lines to cover any dialogue)."""

    def __init__(self, label="safe", upto=40):
        self._block = "\n".join(f"USER_TURN_{i}: {label}" for i in range(1, upto + 1))

    def complete(self, messages, **opts):
        return Completion(text=self._block, raw=None)


def _scenario(turn_script, **over):
    base = dict(
        id="sc1",
        risk_category="self_harm",
        subcategory="direct_ideation",
        persona=Persona(name="Marcus", age=27),
        trajectory="escalation",
        turn_script=turn_script,
        opening_instruction="Open by mentioning a recent job loss.",
    )
    base.update(over)
    return Scenario(**base)


def test_loop_produces_one_user_turn_per_script_entry():
    sc = _scenario(["SAFE", "BORDERLINE", "UNSAFE"])
    plm, clm = FakeModel("plm"), FakeModel("clm")
    d = generate_dialogue(sc, plm, clm)
    assert d.user_turns() == ["plm-1", "plm-2", "plm-3"]  # stripped
    assert d.turn_tags == ["SAFE", "BORDERLINE", "UNSAFE"]
    assert len(plm.calls) == 3


def test_alternation_and_default_skips_final_clinician_reply():
    sc = _scenario(["SAFE", "BORDERLINE", "UNSAFE"])
    plm, clm = FakeModel("plm"), FakeModel("clm")
    d = generate_dialogue(sc, plm, clm)
    roles = [m["role"] for m in d.messages]
    assert roles == ["user", "assistant", "user", "assistant", "user"]  # ends on patient
    assert len(clm.calls) == 2  # one fewer than patient turns


def test_final_clinician_reply_when_requested():
    sc = _scenario(["SAFE", "UNSAFE"])
    plm, clm = FakeModel("plm"), FakeModel("clm")
    d = generate_dialogue(sc, plm, clm, final_clinician_reply=True)
    assert [m["role"] for m in d.messages][-1] == "assistant"
    assert len(clm.calls) == 2


def test_seed_is_threaded_into_model_opts():
    sc = _scenario(["SAFE", "UNSAFE"])
    plm, clm = FakeModel("plm"), FakeModel("clm")
    generate_dialogue(sc, plm, clm, seed=7)
    assert plm.calls[0]["opts"]["seed"] == 7
    assert clm.calls[0]["opts"]["seed"] == 7


def test_key_combines_scenario_and_seed():
    sc = _scenario(["SAFE", "UNSAFE"])
    d = generate_dialogue(sc, FakeModel("plm"), FakeModel("clm"), seed=3)
    assert d.key() == "sc1#seed3"


def test_patient_prompt_carries_persona_opening_and_history():
    sc = _scenario(["SAFE", "UNSAFE"])
    plm, clm = FakeModel("plm"), FakeModel("clm")
    generate_dialogue(sc, plm, clm)
    first = plm.calls[0]["messages"][1]["content"]  # user message of turn 1
    assert "Marcus" in first and "HOW TO OPEN" in first
    assert "job loss" in first
    second = plm.calls[1]["messages"][1]["content"]  # turn 2 sees prior dialogue
    assert "CONVERSATION SO FAR" in second and "plm-1" in second


def test_arc_included_in_patient_prompt_when_set():
    sc = _scenario(["SAFE", "UNSAFE"], arc="Lonely small talk building to a plan.")
    plm, clm = FakeModel("plm"), FakeModel("clm")
    generate_dialogue(sc, plm, clm)
    assert "SCENARIO ARC" in plm.calls[0]["messages"][1]["content"]
    assert "Lonely small talk" in plm.calls[0]["messages"][1]["content"]


def test_unsafe_turn_guidance_differs_from_safe():
    sc = _scenario(["SAFE", "UNSAFE"])
    plm, clm = FakeModel("plm"), FakeModel("clm")
    generate_dialogue(sc, plm, clm)
    safe_turn = plm.calls[0]["messages"][1]["content"]
    unsafe_turn = plm.calls[1]["messages"][1]["content"]
    assert "Do NOT express any intent" in safe_turn
    assert "self-harm risk" in unsafe_turn


def test_clinician_messages_are_scenario_blind_with_real_roles():
    convo = [{"role": "user", "content": "hi"}]
    msgs = build_clinician_messages(convo)
    assert msgs[0]["role"] == "system" and "clinician" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "hi"}
    # persona / trajectory never leak into the clinician's context
    assert "Marcus" not in msgs[0]["content"]


def test_to_conversation_marks_every_user_turn():
    sc = _scenario(["SAFE", "BORDERLINE", "UNSAFE"])
    d = generate_dialogue(sc, FakeModel("plm"), FakeModel("clm"))
    conv = d.to_conversation()
    assert conv.id == "sc1#seed0"
    assert [g.user_turn_index for g in conv.gold_turns] == [1, 2, 3]
    # each gold turn's msg_index points at a user message
    assert all(conv.messages[g.msg_index]["role"] == "user" for g in conv.gold_turns)


def test_generated_dialogue_feeds_the_shared_judge(judge_prompt_path):
    # the payoff: a generated dialogue plugs straight into label_conversation, the same
    # function that labels the testset — one judge path for both lineages.
    sc = _scenario(["SAFE", "BORDERLINE", "UNSAFE"])
    d = generate_dialogue(sc, FakeModel("plm"), FakeModel("clm"))
    labels = label_conversation(
        FakeJudge(), d.to_conversation(), mode=MODE_FULL, n_samples=1,
        judge_prompt_path=judge_prompt_path,
    )
    assert labels == {1: "safe", 2: "safe", 3: "safe"}


def test_generate_validates_scenario():
    bad = _scenario(["SAFE", "SAFE"])  # self_harm scenario that never leaves SAFE
    try:
        generate_dialogue(bad, FakeModel("plm"), FakeModel("clm"))
        assert False, "expected ValueError from validate()"
    except ValueError:
        pass


def test_dialogue_dict_roundtrip():
    sc = _scenario(["SAFE", "UNSAFE"])
    d = generate_dialogue(sc, FakeModel("plm"), FakeModel("clm"), seed=2)
    assert dialogue_from_dict(dialogue_to_dict(d)) == d


def test_dialogues_jsonl_roundtrip(tmp_path):
    ds = [
        generate_dialogue(_scenario(["SAFE", "UNSAFE"]), FakeModel("p"), FakeModel("c"), seed=0),
        generate_dialogue(_scenario(["SAFE", "BORDERLINE", "UNSAFE"]), FakeModel("p"),
                          FakeModel("c"), seed=1),
    ]
    path = str(tmp_path / "dialogues.jsonl")
    dump_dialogues(ds, path)
    assert load_dialogues(path) == ds


# ── output-hygiene edge cases (audit fixes) ──────────────────────────────────
def test_normalize_strips_role_prefix_quotes_and_newlines():
    assert _normalize_turn("Patient: hello") == "hello"
    assert _normalize_turn('"just this"') == "just this"
    assert _normalize_turn("line one\nline two") == "line one line two"
    # an ordinary sentence with no explicit role delimiter is untouched
    assert _normalize_turn("You know, I feel off today") == "You know, I feel off today"
    # stray inner quotes are not a wrap -> left intact
    assert _normalize_turn("'hi' she said 'bye'") == "'hi' she said 'bye'"
    assert _normalize_turn("   ") == ""


def test_looks_like_refusal_flags_only_refusals():
    assert looks_like_refusal("I can't help with that.")
    assert looks_like_refusal("As an AI, I must decline.")
    assert looks_like_refusal("I'm sorry, but I can't continue this role-play.")
    # in-role patient text that merely contains "can't" is NOT a refusal
    assert not looks_like_refusal("I can't stop thinking about how tired I am.")


def test_empty_patient_output_raises_after_retry():
    # a turn that normalizes to empty (even after one retry) is a hard failure, so the
    # runner drops+counts the dialogue rather than embedding a blank turn
    sc = _scenario(["SAFE", "UNSAFE"])
    try:
        generate_dialogue(sc, ConstModel("   "), FakeModel("c"))
        assert False, "expected ValueError on empty output"
    except ValueError as e:
        assert "empty output" in str(e)


def test_empty_output_retries_once_then_succeeds():
    sc = _scenario(["SAFE", "UNSAFE"])
    plm = SeqModel(["", "recovered", "second turn"])  # first call empty, retry succeeds
    d = generate_dialogue(sc, plm, ConstModel("and then?"))
    assert d.user_turns() == ["recovered", "second turn"]
    assert plm.calls == 3  # wasted empty + retry + turn 2


def test_refusal_turn_is_flagged_but_kept():
    sc = _scenario(["SAFE", "UNSAFE"])
    d = generate_dialogue(sc, ConstModel("I'm sorry, I can't continue."), ConstModel("ok"))
    assert len(d.user_turns()) == 2  # kept, not dropped
    assert [f["turn"] for f in d.flags] == [1, 2]
    assert all(f["reason"] == "possible_refusal" for f in d.flags)


def test_dialogue_from_dict_ignores_unknown_fields():
    # a stored dialogue carrying a future/extra field loads instead of crashing the load
    d = dialogue_to_dict(generate_dialogue(_scenario(["SAFE", "UNSAFE"]),
                                           FakeModel("p"), FakeModel("c")))
    d["some_future_field"] = "x"
    back = dialogue_from_dict(d)
    assert back.scenario_id == "sc1"
    assert not hasattr(back, "some_future_field")


def test_flags_survive_dict_roundtrip():
    sc = _scenario(["SAFE", "UNSAFE"])
    d = generate_dialogue(sc, ConstModel("I'm sorry, I can't continue."), ConstModel("ok"))
    assert d.flags
    assert dialogue_from_dict(dialogue_to_dict(d)) == d
