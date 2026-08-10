# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Scenario schema: validation, serialization, and the example library."""

from mh_guard_harness.scenarios import (
    Persona,
    Scenario,
    errors,
    from_dict,
    load_library,
    to_dict,
    validate,
    validate_library,
)


def _safe(**over):
    base = dict(
        id="s1",
        risk_category="safe",
        subcategory="work",
        persona=Persona(name="Dana"),
        trajectory="de_escalation",
        turn_script=["SAFE", "SAFE"],
        opening_instruction="Open about work stress.",
    )
    base.update(over)
    return Scenario(**base)


def test_valid_safe_scenario_has_no_errors():
    assert errors(_safe()) == []
    validate(_safe())  # does not raise


def test_max_turns_tracks_turn_script():
    assert _safe(turn_script=["SAFE", "SAFE", "SAFE"]).max_turns == 3


def test_bad_risk_category():
    assert any("risk_category" in e for e in errors(_safe(risk_category="nope")))


def test_subcategory_must_match_risk_category():
    # 'violent_ideation' is a harm_to_others subcategory, invalid under 'self_harm'
    s = _safe(risk_category="self_harm", subcategory="violent_ideation",
              turn_script=["SAFE", "UNSAFE"])
    assert any("subcategory" in e for e in errors(s))


def test_safe_scenario_cannot_have_unsafe_turns():
    assert any("UNSAFE" in e for e in errors(_safe(turn_script=["SAFE", "UNSAFE"])))


def test_safe_scenario_may_touch_borderline():
    # a safe conversation that wavers into ambiguity but never crosses into UNSAFE is a
    # valid hard-negative — not an authoring error
    assert errors(_safe(turn_script=["SAFE", "BORDERLINE", "SAFE"])) == []


def test_unsafe_scenario_must_reach_risk():
    # a self_harm scenario that never leaves SAFE is an authoring error
    s = _safe(risk_category="self_harm", subcategory="passive_ideation",
              turn_script=["SAFE", "SAFE"])
    assert any("never leaves SAFE" in e for e in errors(s))


def test_borderline_alone_satisfies_unsafe_scenario():
    s = _safe(risk_category="harm_to_others", subcategory="violent_ideation",
              trajectory="sustained_ambiguity", turn_script=["SAFE", "BORDERLINE"])
    assert errors(s) == []


def test_empty_turn_script_and_bad_tags():
    assert any("turn_script is empty" in e for e in errors(_safe(turn_script=[])))
    assert any("invalid tags" in e for e in errors(_safe(turn_script=["SAFE", "MAYBE"])))


def test_empty_opening_flagged():
    assert any("opening_instruction" in e for e in errors(_safe(opening_instruction="")))


def test_roundtrip_dict_preserves_persona():
    s = _safe(persona=Persona(name="Marcus", age=27, backstory="You felt low."))
    back = from_dict(to_dict(s))
    assert back == s
    assert back.persona.name == "Marcus" and back.persona.age == 27


def test_from_dict_ignores_unknown_fields():
    # a model embellishes with fields the schema lacks -> dropped, not a crash
    d = to_dict(_safe())
    d["persona"]["additional_notes"] = "extra clinical caveat"
    d["some_future_field"] = "x"
    s = from_dict(d)
    assert s.persona.name == "Dana"
    assert not hasattr(s.persona, "additional_notes")


def test_validate_library_rejects_duplicate_ids():
    lib = [_safe(id="dup"), _safe(id="dup")]
    try:
        validate_library(lib)
        assert False, "expected ValueError on duplicate id"
    except ValueError as e:
        assert "duplicate" in str(e)


def test_validate_library_reports_balance_and_drift():
    lib = [
        _safe(id="a"),
        _safe(id="b", risk_category="self_harm", subcategory="direct_ideation",
              turn_script=["SAFE", "UNSAFE"]),
    ]
    out = validate_library(lib)
    assert out["n"] == 2
    assert out["balance"]["safe"] == 0.5
    assert set(out["drift"]) == {"safe", "self_harm", "harm_to_others"}


def test_example_library_loads_and_validates():
    lib = load_library("data/scenarios/examples.json")
    assert len(lib) == 4
    summary = validate_library(lib)  # raises if any example is malformed
    assert summary["n"] == 4
    # covers all three risk categories
    assert set(summary["counts"]) == {"safe", "self_harm", "harm_to_others"}
