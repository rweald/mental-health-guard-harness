# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Decision-model scoring: spec validation, state, answer -> score, run/resume/derive.

The spec fixture is obviously synthetic — the real spec is user-supplied and never
committed, like the guard and judge prompts.
"""

import json

import pytest

from mh_guard_harness import decisions, metrics
from mh_guard_harness.models import Decision, DecisionAPIError

FIXTURE_SPEC = {
    "spec_version": 1,
    "questions": {
        "risk_category": {
            "type": "choice",
            "instructions": "FIXTURE choice instructions",
            "criteria": {
                "safe": "FIXTURE option A",
                "self_harm": "FIXTURE option B",
                "harm_to_others": "FIXTURE option C",
            },
        },
        "unsafe": {
            "type": "noul",
            "instructions": "FIXTURE yes/no instructions",
            "criteria": {"true": "FIXTURE yes", "false": "FIXTURE no"},
        },
    },
}


def _write_spec(tmp_path, spec=FIXTURE_SPEC, name="spec.json"):
    path = tmp_path / name
    path.write_text(json.dumps(spec))
    return path


def _spec_with(mutate):
    spec = json.loads(json.dumps(FIXTURE_SPEC))
    mutate(spec)
    return spec


# --- spec ----------------------------------------------------------------------------


def test_load_spec_accepts_fixture(tmp_path):
    spec = decisions.load_spec(_write_spec(tmp_path))
    assert set(spec.questions) == {"risk_category", "unsafe"}
    assert spec.filename == "spec.json"
    assert len(spec.sha256) == 64


def test_spec_hash_ignores_formatting_but_not_text(tmp_path):
    a = decisions.load_spec(_write_spec(tmp_path, name="a.json"))
    pretty = tmp_path / "b.json"
    pretty.write_text(json.dumps(FIXTURE_SPEC, indent=4, sort_keys=True))
    assert decisions.load_spec(pretty).sha256 == a.sha256

    def edit(s):
        s["questions"]["unsafe"]["instructions"] += "!"

    changed = decisions.load_spec(_write_spec(tmp_path, _spec_with(edit), name="c.json"))
    assert changed.sha256 != a.sha256


@pytest.mark.parametrize("mutate, match", [
    (lambda s: s.update(spec_version=2), "spec_version"),
    (lambda s: s["questions"].pop("unsafe"), "questions must be exactly"),
    (lambda s: s["questions"]["unsafe"].update(type="choice"), "type must be"),
    (lambda s: s["questions"]["unsafe"].update(instructions="  "), "instructions"),
    (lambda s: s["questions"]["risk_category"]["criteria"].pop("self_harm"), "criteria keys"),
    (lambda s: s["questions"]["unsafe"]["criteria"].update(true=""), "non-empty"),
    (lambda s: s["questions"]["unsafe"].update(extra=1), "exactly type"),
])
def test_load_spec_rejects_structural_errors(tmp_path, mutate, match):
    with pytest.raises(decisions.SpecError, match=match):
        decisions.load_spec(_write_spec(tmp_path, _spec_with(mutate)))


def test_spec_errors_never_quote_spec_text(tmp_path):
    def mutate(s):
        s["questions"]["unsafe"]["criteria"]["true"] = ""

    with pytest.raises(decisions.SpecError) as e:
        decisions.load_spec(_write_spec(tmp_path, _spec_with(mutate)))
    assert "FIXTURE" not in str(e.value)


def test_load_spec_requires_a_path():
    with pytest.raises(decisions.SpecError, match="--spec"):
        decisions.load_spec(None)


def test_load_spec_bad_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(decisions.SpecError, match="not valid JSON"):
        decisions.load_spec(path)


# --- state ---------------------------------------------------------------------------


def test_build_state_separates_target_turn_from_history():
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    assert decisions.build_state(msgs) == {
        "conversation_history": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ],
        "message_to_assess": "u2",
    }


def test_build_state_first_turn_has_empty_history():
    state = decisions.build_state([{"role": "user", "content": "hi"}])
    assert state["conversation_history"] == []


def test_build_state_rejects_row_not_ending_on_user():
    with pytest.raises(ValueError, match="final message"):
        decisions.build_state([{"role": "user", "content": "u"},
                               {"role": "assistant", "content": "a"}])


# --- answer -> score -----------------------------------------------------------------


def _choice(probs, **kw):
    return {"risk_category": {"type": "choice", "choice": "safe", "confidence": 0.9,
                              "probabilities": probs, **kw}}


def test_choice3_score_is_one_minus_p_safe():
    s = decisions.score_choice3(_choice({"safe": 0.7, "self_harm": 0.2,
                                         "harm_to_others": 0.1}))
    assert s.failure is None
    assert s.p_unsafe == pytest.approx(0.3)
    assert s.extra["probabilities"]["self_harm"] == pytest.approx(0.2)
    assert s.extra["choice"] == "safe"


def test_choice3_renormalizes_unnormalized_probabilities():
    s = decisions.score_choice3(_choice({"safe": 0.4, "self_harm": 0.4,
                                         "harm_to_others": 0.0}))
    assert s.p_unsafe == pytest.approx(0.5)


@pytest.mark.parametrize("probs", [
    {"safe": 0.5, "self_harm": 0.5},                               # missing option
    {"safe": 0.5, "self_harm": 0.5, "harm_to_others": 0.0, "x": 0},  # extra option
    {"safe": 1.5, "self_harm": 0.0, "harm_to_others": 0.0},        # out of range
    {"safe": float("nan"), "self_harm": 0.5, "harm_to_others": 0.5},
    {"safe": 0.0, "self_harm": 0.0, "harm_to_others": 0.0},        # no mass
])
def test_choice3_invalid_probabilities_are_failures_not_clamped(probs):
    s = decisions.score_choice3(_choice(probs))
    assert s.failure == decisions.INVALID
    assert s.p_unsafe is None


def test_yesno_score_is_returned_probability():
    s = decisions.score_yesno({"unsafe": {"type": "noul", "noul": 0.83}})
    assert s.p_unsafe == pytest.approx(0.83)


@pytest.mark.parametrize("answer", [None, {"type": "noul"}, {"type": "noul", "noul": 1.2},
                                    {"type": "choice", "noul": 0.5}])
def test_yesno_invalid(answer):
    answers = {} if answer is None else {"unsafe": answer}
    assert decisions.score_yesno(answers).failure == decisions.INVALID


def test_refusal_is_its_own_failure_kind():
    answers = {"unsafe": {"type": "refusal", "refusal": "declined"},
               "risk_category": {"type": "refusal"}}
    assert decisions.score_yesno(answers).failure == decisions.REFUSAL
    assert decisions.score_choice3(answers).failure == decisions.REFUSAL


# --- run / resume / derive -----------------------------------------------------------


def _rows(n):
    return [
        {"index": i, "prompt": [{"role": "user", "content": f"turn {i}"}],
         "label": "safe" if i % 2 else "unsafe_self_harm_risk",
         "annotator_labels": {"annotator_1": "safe"}}
        for i in range(n)
    ]


def _answers(p):
    return {
        "risk_category": {"type": "choice", "choice": "safe", "confidence": 0.5,
                          "probabilities": {"safe": 1 - p, "self_harm": p,
                                            "harm_to_others": 0.0}},
        "unsafe": {"type": "noul", "noul": p},
    }


class FakeDecider:
    """Answers from a per-row script keyed on the state text; raises for listed rows."""

    def __init__(self, fail=(), refuse=()):
        self.fail, self.refuse, self.calls = set(fail), set(refuse), []

    def decide(self, state, questions):
        i = int(state["message_to_assess"].split()[-1])
        self.calls.append(i)
        if i in self.fail:
            raise DecisionAPIError("HTTP 502: down", status=502)
        if i in self.refuse:
            answers = {"risk_category": {"type": "refusal"}, "unsafe": {"type": "refusal"}}
        else:
            answers = _answers(0.9 if i % 2 == 0 else 0.1)
        return Decision(answers=answers, raw={
            "model": "vendor/m-20261001", "provider": "Vendor", "answers": answers,
            "usage": {"input_tokens": 10, "output_tokens": 0, "cost": 0.001},
        })


def _run_and_derive(tmp_path, model, rows, spec):
    cache = tmp_path / "m.responses.jsonl"
    decisions.run(model, rows, spec, cache, model_id="vendor/m", workers=2)
    return cache, decisions.derive(
        rows, cache, tmp_path, key="m", spec=spec, model_id="vendor/m",
        provenance_base={"model": "vendor/m"},
    )


def test_run_and_derive_write_both_seam_files(tmp_path):
    spec = decisions.load_spec(_write_spec(tmp_path))
    rows = _rows(6)
    _, summary = _run_and_derive(tmp_path, FakeDecider(), rows, spec)

    for suffix in ("choice3", "yesno"):
        path = tmp_path / f"m__{suffix}_scores.jsonl"
        df = metrics.load_scores(path)  # the existing metrics read it unchanged
        assert sorted(df["index"]) == list(range(6))
        assert metrics.auroc(df["y_true"], df["p_unsafe"]) == 1.0
        prov = json.loads(path.with_suffix(".provenance.json").read_text())
        assert prov["decision_spec"]["sha256"] == spec.sha256
        assert prov["run"]["resolved_models"] == {"vendor/m-20261001": 6}
        assert prov["run"]["cost_usd"] == pytest.approx(0.006)
    assert summary["run"]["unanswered"] == 0


def test_failures_and_refusals_are_listed_not_scored(tmp_path):
    spec = decisions.load_spec(_write_spec(tmp_path))
    rows = _rows(6)
    _, summary = _run_and_derive(tmp_path, FakeDecider(fail={1}, refuse={2}), rows, spec)

    assert summary["questions"]["unsafe"] == {
        "path": str(tmp_path / "m__yesno_scores.jsonl"), "scored": 4,
        "error": 1, "refusal": 1,
    }
    failures = [json.loads(x) for x in (tmp_path / "m.failures.jsonl").read_text().splitlines()]
    assert {(f["index"], f["kind"]) for f in failures} == {
        (1, "error"), (2, "refusal"),
    }
    assert len(failures) == 4  # each listed once per question


def test_rerun_retries_only_failed_rows(tmp_path):
    spec = decisions.load_spec(_write_spec(tmp_path))
    rows = _rows(4)
    cache = tmp_path / "m.responses.jsonl"
    decisions.run(FakeDecider(fail={1}, refuse={2}), rows, spec, cache, model_id="vendor/m")

    second = FakeDecider()
    made = decisions.run(second, rows, spec, cache, model_id="vendor/m")
    assert made == 1 and second.calls == [1]  # refusal at row 2 counts as answered

    summary = decisions.derive(rows, cache, tmp_path, key="m", spec=spec,
                               model_id="vendor/m", provenance_base={})
    assert summary["run"]["unanswered"] == 0


def test_cache_refuses_a_different_spec(tmp_path):
    spec = decisions.load_spec(_write_spec(tmp_path))
    rows = _rows(2)
    cache = tmp_path / "m.responses.jsonl"
    decisions.run(FakeDecider(), rows, spec, cache, model_id="vendor/m")

    def edit(s):
        s["questions"]["unsafe"]["instructions"] = "FIXTURE changed"

    other = decisions.load_spec(_write_spec(tmp_path, _spec_with(edit), name="other.json"))
    with pytest.raises(decisions.CacheMismatchError, match="spec"):
        decisions.run(FakeDecider(), rows, other, cache, model_id="vendor/m")


def test_cache_refuses_a_different_model(tmp_path):
    spec = decisions.load_spec(_write_spec(tmp_path))
    cache = tmp_path / "m.responses.jsonl"
    decisions.run(FakeDecider(), _rows(2), spec, cache, model_id="vendor/m")
    with pytest.raises(decisions.CacheMismatchError, match="model"):
        decisions.read_cache(cache, spec_sha256=spec.sha256, model_id="vendor/other")


def test_derive_marks_unrun_rows_unanswered(tmp_path):
    spec = decisions.load_spec(_write_spec(tmp_path))
    rows = _rows(4)
    cache = tmp_path / "m.responses.jsonl"
    decisions.run(FakeDecider(), rows[:2], spec, cache, model_id="vendor/m")
    summary = decisions.derive(rows, cache, tmp_path, key="m", spec=spec,
                               model_id="vendor/m", provenance_base={})
    assert summary["run"]["unanswered"] == 2
    assert summary["questions"]["unsafe"]["scored"] == 2
