# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Scenario-authoring harness: distribution quota, parsing, and per-batch validation.
No paid calls — a fake model returns canned JSON."""

import json
from collections import Counter

from mh_guard_harness.authoring import (
    Cell,
    _apportion,
    author_batch,
    build_cells,
    parse_json_scenarios,
    recover_rejects,
)
from mh_guard_harness.models import Completion
from mh_guard_harness.scenarios import SUBCATEGORIES


class FakeModel:
    def __init__(self, text):
        self._text = text

    def complete(self, messages, **opts):
        return Completion(text=self._text, raw=None)


def test_apportion_sums_exactly():
    out = _apportion(100, {"a": 0.62, "b": 0.21, "c": 0.17})
    assert sum(out.values()) == 100
    assert out == {"a": 62, "b": 21, "c": 17}


def test_apportion_largest_remainder_distributes_leftovers():
    out = _apportion(10, {"a": 1, "b": 1, "c": 1})  # 3.33 each
    assert sum(out.values()) == 10
    assert sorted(out.values()) == [3, 3, 4]


def test_build_cells_hits_total_and_paper_balance():
    cells = build_cells(100)
    assert len(cells) == 100
    cats = Counter(c.risk_category for c in cells)
    assert cats == {"safe": 62, "self_harm": 21, "harm_to_others": 17}


def test_build_cells_overweights_borderline_subcategories():
    cells = build_cells(1000)
    sub = Counter(c.subcategory for c in cells if c.risk_category == "self_harm")
    # passive_ideation is weighted 2.5x vs method_inquiry 1.0x
    assert sub["passive_ideation"] > sub["method_inquiry"]


def test_build_cells_only_emits_valid_combinations():
    for c in build_cells(200):
        assert c.subcategory in SUBCATEGORIES[c.risk_category]


def test_parse_plain_array():
    assert parse_json_scenarios('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]


def test_parse_strips_json_fence():
    fenced = '```json\n[{"a": 1}]\n```'
    assert parse_json_scenarios(fenced) == [{"a": 1}]


def test_parse_single_object_becomes_list():
    assert parse_json_scenarios('{"a": 1}') == [{"a": 1}]


def test_parse_falls_back_to_outermost_array_amid_prose():
    text = 'Here you go:\n[{"a": 1}]\nHope that helps!'
    assert parse_json_scenarios(text) == [{"a": 1}]


def test_parse_garbage_returns_empty():
    assert parse_json_scenarios("not json at all") == []


def test_parse_tolerates_trailing_comma():
    # the exact 'missing' failure: a valid array with a trailing comma before a brace
    reply = '```json\n[{"a": 1,}, {"b": 2},]\n```'
    assert parse_json_scenarios(reply) == [{"a": 1}, {"b": 2}]


def test_parse_tolerates_dangling_bare_string_member():
    # a bare "" (or any bare string) just before a closing brace is an object member with
    # no key -> stripped, not a parse failure. A real "key":"value"} is untouched.
    reply = '[{"name": "X", "backstory": "long text", ""}, {"k": "v"}]'
    assert parse_json_scenarios(reply) == [{"name": "X", "backstory": "long text"}, {"k": "v"}]


def test_parse_object_level_recovers_around_one_broken_sibling():
    # one malformed object among good ones no longer drops the whole batch; the broken
    # slot becomes an empty-dict placeholder so positions stay aligned to cells
    reply = '[{"a": 1}, {"b": 2, "c": }, {"c": 3}]'  # middle object has a valueless key
    out = parse_json_scenarios(reply)
    assert out == [{"a": 1}, {}, {"c": 3}]


def test_scenarios_from_reply_treats_placeholder_as_missing():
    from mh_guard_harness.authoring import scenarios_from_reply

    cells = [Cell("self_harm", "direct_ideation", "escalation"),
             Cell("safe", "grief", "de_escalation")]
    # first object is good; second is a broken sibling -> placeholder -> 'missing'
    good = json.dumps(_obj("self_harm", ["SAFE", "UNSAFE"]))[1:-1]  # inner fields
    reply = "[{" + good + '}, {"persona": {"name": "X"}, "bad": }]'  # valueless key
    res = scenarios_from_reply(reply, cells)
    assert len(res.scenarios) == 1 and res.scenarios[0].risk_category == "self_harm"
    assert len(res.rejects) == 1 and "missing" in res.rejects[0]["reason"]


def _obj(risk_category, turn_script):
    # a model reply object; risk_category here is intentionally the model's (to be
    # overridden by the cell's assignment inside author_batch)
    return {
        "risk_category": risk_category,
        "subcategory": "whatever",
        "trajectory": "whatever",
        "persona": {"name": "Test", "age": 30},
        "turn_script": turn_script,
        "opening_instruction": "Open naturally.",
        "arc": "A short arc.",
    }


def test_author_batch_restamps_distribution_and_validates():
    cells = [
        Cell("self_harm", "passive_ideation", "escalation"),
        Cell("safe", "work", "de_escalation"),
    ]
    # model returns wrong risk_category/subcategory/trajectory on purpose
    reply = json.dumps([
        _obj("harm_to_others", ["SAFE", "BORDERLINE", "UNSAFE"]),
        _obj("self_harm", ["SAFE", "SAFE", "SAFE"]),
    ])
    res = author_batch(FakeModel(reply), cells)
    out = res.scenarios
    assert len(out) == 2 and res.rejects == []
    # the cell's assignment wins, not the model's
    assert out[0].risk_category == "self_harm" and out[0].subcategory == "passive_ideation"
    assert out[1].risk_category == "safe" and out[1].trajectory == "de_escalation"
    assert [s.id for s in out] == ["b000_00", "b000_01"]


def test_author_batch_rejects_schema_invalid_with_reason():
    # safe cell, but the model's turn_script has an UNSAFE turn -> after re-stamping
    # risk_category=safe, validate() rejects it (safe can't be unsafe)
    cells = [Cell("safe", "work", "de_escalation")]
    reply = json.dumps([_obj("safe", ["SAFE", "UNSAFE"])])
    res = author_batch(FakeModel(reply), cells)
    assert res.scenarios == []
    assert len(res.rejects) == 1
    assert "UNSAFE" in res.rejects[0]["reason"]
    assert res.rejects[0]["cell"]["risk_category"] == "safe"


def test_author_batch_captures_raw_and_missing_cells():
    cells = [Cell("self_harm", "direct_ideation", "escalation"),
             Cell("safe", "grief", "sustained_ambiguity")]
    reply = json.dumps([_obj("self_harm", ["SAFE", "UNSAFE"])])  # only one object
    res = author_batch(FakeModel(reply), cells)
    assert len(res.scenarios) == 1 and res.scenarios[0].risk_category == "self_harm"
    # the second cell had no object -> recorded as a 'missing' reject, raw kept
    assert len(res.rejects) == 1 and "missing" in res.rejects[0]["reason"]
    assert res.raw == reply


def test_author_batch_tolerates_model_added_persona_fields():
    # the exact failure seen in the smoke: Opus added persona.additional_notes, which
    # crashed Persona(**persona) and wiped out every risk scenario
    cells = [Cell("self_harm", "passive_ideation", "escalation")]
    obj = _obj("self_harm", ["SAFE", "BORDERLINE", "UNSAFE"])
    obj["persona"]["additional_notes"] = "clinician caveat"
    res = author_batch(FakeModel(json.dumps([obj])), cells)
    assert len(res.scenarios) == 1 and res.rejects == []


def test_author_batch_refusal_rejects_all_cells():
    # a model that refuses (no JSON) drops every cell, but we can see the raw reply
    cells = [Cell("self_harm", "direct_ideation", "escalation")]
    res = author_batch(FakeModel("I can't help with that."), cells)
    assert res.scenarios == []
    assert len(res.rejects) == 1
    assert res.raw == "I can't help with that."


def test_recover_rejects_salvages_parser_and_relaxed_rule():
    # a whole batch that failed to parse (trailing comma) and a safe-with-BORDERLINE that
    # the old rule wrongly rejected both come back; a safe-with-UNSAFE mislabel does not.
    safe_border = _obj("safe", ["SAFE", "BORDERLINE", "SAFE"])
    safe_unsafe = _obj("safe", ["SAFE", "UNSAFE"])
    raw = json.dumps([safe_border, safe_unsafe])[:-1] + ",]"  # trailing comma broke the parse
    payload = {
        "rejects": [
            {"batch_index": 0, "index": 0, "cell": vars(Cell("safe", "venting", "sustained_ambiguity")),
             "reason": "missing: model returned fewer objects than cells"},
            {"batch_index": 0, "index": 1, "cell": vars(Cell("safe", "venting", "de_escalation")),
             "reason": "missing: model returned fewer objects than cells"},
        ],
        "raw_by_batch": {"0": raw},
    }
    recovered, still = recover_rejects(payload)
    assert len(recovered) == 1  # only the safe-with-BORDERLINE survives re-validation
    assert recovered[0].risk_category == "safe" and recovered[0].subcategory == "venting"
    assert recovered[0].turn_script == ["SAFE", "BORDERLINE", "SAFE"]
    assert len(still) == 1 and "UNSAFE" in still[0]["recover_reason"]


def test_recover_rejects_skips_records_without_index():
    # a batch-level failure (no per-cell index/cell) can't be re-parsed; it stays rejected
    payload = {"rejects": [{"batch_index": 3, "reason": "batch call failed: timeout"}],
               "raw_by_batch": {}}
    recovered, still = recover_rejects(payload)
    assert recovered == [] and len(still) == 1
