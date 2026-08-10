# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""The prompt loader: modes, hash enforcement, normalization, and provenance.

The invariant behind most of this file is that the harness ships no prompt text and
must not silently substitute one. So: paths are explicit, a missing file is an error
rather than a fallback, and reproduction mode fails loudly on anything but an exact
(normalized) match.

Note there is no test that the real reference prompts load — that would require
committing them, which is exactly what this design avoids. The hash-match path is
tested against a synthetic spec registered for the test instead, which exercises the
same code with content we are free to ship.
"""

import subprocess
import sys
import unicodedata

import pytest

from mh_guard_harness import paper_prompts
from mh_guard_harness.paper_prompts import (
    MODE_CUSTOM,
    MODE_REPRODUCTION,
    PROMPT_SPECS,
    PromptHashMismatchError,
    PromptPathError,
    PromptPlaceholderError,
    PromptSpec,
    load,
    normalize,
    provenance,
    render,
    sha256,
)

FIXTURE_TEXT = "FIXTURE PROMPT\n<<SLOT>>\n{conversation}\n<</SLOT>>"


def _write(tmp_path, text, name="p.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- the package must be importable with no prompt files anywhere ----------------


def test_package_imports_with_no_prompt_files_present():
    """The loader is lazy: importing the package must not touch the filesystem, so a
    fresh checkout with no prompts still imports (and this suite still runs)."""
    code = (
        "import mh_guard_harness as m;"
        "from mh_guard_harness import judging, paper_prompts;"
        "from mh_guard_harness.modeling import scoring;"
        "from mh_guard_harness.training import dataset;"
        "print('ok')"
    )
    # cwd=/ so nothing resolves relative to the repo, and no prompts/ is in reach.
    r = subprocess.run(
        [sys.executable, "-c", code], cwd="/", capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


# --- paths are explicit ----------------------------------------------------------


def test_unset_path_names_the_config_key_and_cli_flag():
    with pytest.raises(PromptPathError) as e:
        load("guard", path=None)
    msg = str(e.value)
    assert "guard_prompt_path" in msg and "--guard-prompt" in msg


def test_missing_file_reports_the_path_the_figure_and_the_paper_url(tmp_path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(PromptPathError) as e:
        load("guard", path=missing)
    msg = str(e.value)
    assert str(missing) in msg
    assert PROMPT_SPECS["guard"].figure in msg
    assert paper_prompts.PAPER_URL in msg


def test_loads_from_a_cwd_outside_the_repo(tmp_path, monkeypatch):
    """An absolute prompt path must work from anywhere — no implicit CWD dependency."""
    path = _write(tmp_path, FIXTURE_TEXT)
    monkeypatch.chdir("/")
    assert load("guard", path=path).text == FIXTURE_TEXT


def test_unknown_prompt_name_lists_the_known_ones(tmp_path):
    with pytest.raises(KeyError) as e:
        load("nonesuch", path=_write(tmp_path, FIXTURE_TEXT))
    assert "guard" in str(e.value) and "judge" in str(e.value)


def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown mode"):
        load("guard", path=_write(tmp_path, FIXTURE_TEXT), mode="sideways")


# --- placeholder must appear exactly once ----------------------------------------


@pytest.mark.parametrize(
    "text, found",
    [
        ("no placeholder at all", 0),
        ("{conversation} and again {conversation}", 2),
    ],
)
def test_placeholder_must_appear_exactly_once(tmp_path, text, found):
    with pytest.raises(PromptPlaceholderError) as e:
        load("guard", path=_write(tmp_path, text))
    assert "{conversation}" in str(e.value)
    assert f"found it {found} times" in str(e.value)


def test_judge_placeholder_is_transcript(tmp_path):
    with pytest.raises(PromptPlaceholderError) as e:
        load("judge", path=_write(tmp_path, "{conversation} is the wrong slot"))
    assert "{transcript}" in str(e.value)


# --- custom vs reproduction ------------------------------------------------------


def test_custom_mode_accepts_a_prompt_that_does_not_match_the_reference(tmp_path):
    """The whole point of custom mode: bring your own prompt. The hash is recorded,
    never compared."""
    loaded = load("guard", path=_write(tmp_path, FIXTURE_TEXT), mode=MODE_CUSTOM)
    assert loaded.sha256 != PROMPT_SPECS["guard"].sha256
    assert loaded.hash_enforced is False
    assert loaded.source == paper_prompts.SOURCE_CUSTOM
    assert loaded.mode == MODE_CUSTOM


def test_custom_is_the_default_mode(tmp_path):
    assert load("guard", path=_write(tmp_path, FIXTURE_TEXT)).mode == MODE_CUSTOM


def test_reproduction_mode_rejects_a_prompt_that_is_not_the_reference(tmp_path):
    with pytest.raises(PromptHashMismatchError) as e:
        load("guard", path=_write(tmp_path, FIXTURE_TEXT), mode=MODE_REPRODUCTION)
    msg = str(e.value)
    assert PROMPT_SPECS["guard"].sha256 in msg      # what was expected
    assert sha256(FIXTURE_TEXT) in msg              # what was found
    assert FIXTURE_TEXT not in msg                  # but never the content itself


@pytest.fixture
def synthetic_spec(monkeypatch):
    """Register a spec whose reference hash is the fixture text's.

    Testing the hash-match path against the real reference prompts would mean
    committing them; a synthetic spec exercises identical code with shippable content.
    """
    spec = PromptSpec(
        name="fixture",
        placeholder="{conversation}",
        figure="Figure 0 (test fixture)",
        sha256=sha256(normalize(FIXTURE_TEXT)),
        role="synthetic spec used only by the test suite",
    )
    monkeypatch.setitem(PROMPT_SPECS, "fixture", spec)
    return spec


def test_reproduction_mode_accepts_the_matching_prompt(tmp_path, synthetic_spec):
    loaded = load("fixture", path=_write(tmp_path, FIXTURE_TEXT), mode=MODE_REPRODUCTION)
    assert loaded.sha256 == synthetic_spec.sha256
    assert loaded.hash_enforced is True
    assert loaded.source == paper_prompts.SOURCE_REFERENCE


def test_reproduction_mode_forgives_whitespace_only_transcription_noise(
    tmp_path, synthetic_spec
):
    """Transcribing out of a PDF picks up CRLF and trailing spaces; those must not
    fail the check, since they do not change the prompt."""
    noisy = FIXTURE_TEXT.replace("\n", "  \r\n") + "\n\n"
    loaded = load("fixture", path=_write(tmp_path, noisy), mode=MODE_REPRODUCTION)
    assert loaded.sha256 == synthetic_spec.sha256


def test_matching_content_reports_reference_source_even_in_custom_mode(
    tmp_path, synthetic_spec
):
    """Provenance should say the reference prompt was used even when the run did not
    ask for it to be enforced — but must not claim it was verified."""
    loaded = load("fixture", path=_write(tmp_path, FIXTURE_TEXT), mode=MODE_CUSTOM)
    assert loaded.source == paper_prompts.SOURCE_REFERENCE
    assert loaded.hash_enforced is False


# --- no caching / no cross-contamination -----------------------------------------


def test_two_different_custom_prompts_do_not_cross_contaminate(tmp_path):
    """A cache keyed on anything less than (path, mode) would return the first
    prompt for the second file."""
    a = _write(tmp_path, "PROMPT A {conversation}", name="a.txt")
    b = _write(tmp_path, "PROMPT B {conversation}", name="b.txt")
    la, lb = load("guard", path=a), load("guard", path=b)
    assert la.text != lb.text
    assert la.sha256 != lb.sha256
    assert load("guard", path=a).text == la.text  # and re-reading a still gives a


def test_edits_to_a_prompt_file_are_picked_up(tmp_path):
    """No caching means an edited file takes effect on the next call."""
    path = _write(tmp_path, "FIRST {conversation}")
    first = load("guard", path=path).sha256
    path.write_text("SECOND {conversation}", encoding="utf-8")
    assert load("guard", path=path).sha256 != first


# --- normalization goldens -------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("a\r\nb", "a\nb"),                      # CRLF
        ("a\rb", "a\nb"),                        # lone CR (old Mac)
        ("a   \nb\t\t\n", "a\nb"),               # trailing spaces and tabs, per line
        ("a\n\n\n", "a"),                        # multiple trailing newlines
        ("a\n\n\nb", "a\n\n\nb"),                # interior blank lines are content
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_goldens(raw, expected):
    assert normalize(raw) == expected


def test_normalize_composes_nfd_to_nfc():
    nfd = unicodedata.normalize("NFD", "café")
    assert nfd != "café"                          # precondition: really decomposed
    assert normalize(nfd) == "café"
    assert sha256(normalize(nfd)) == sha256(normalize("café"))


def test_normalize_is_idempotent():
    raw = "a  \r\nb\t\n\n" + unicodedata.normalize("NFD", "é")
    once = normalize(raw)
    assert normalize(once) == once


def test_normalize_preserves_interior_whitespace():
    assert normalize("a  b\n  indented") == "a  b\n  indented"


def test_loaded_text_is_the_normalized_text(tmp_path):
    loaded = load("guard", path=_write(tmp_path, "x  \r\n{conversation}\n\n"))
    assert loaded.text == "x\n{conversation}"
    assert loaded.sha256 == sha256(loaded.text)


# --- rendering -------------------------------------------------------------------


def test_render_substitutes_the_placeholder(tmp_path):
    loaded = load("guard", path=_write(tmp_path, FIXTURE_TEXT))
    assert render(loaded, "HELLO") == FIXTURE_TEXT.replace("{conversation}", "HELLO")


def test_render_tolerates_braces_in_the_substituted_value(tmp_path):
    """Substitution is str.replace, not str.format — conversation text containing
    braces must not blow up or be interpreted."""
    loaded = load("guard", path=_write(tmp_path, FIXTURE_TEXT))
    assert "{not_a_field}" in render(loaded, "User: {not_a_field}")


def test_render_tolerates_other_braces_in_the_prompt(tmp_path):
    """A user's own prompt may contain braces for unrelated reasons."""
    loaded = load("guard", path=_write(tmp_path, 'use {"json": true} then {conversation}'))
    assert render(loaded, "X") == 'use {"json": true} then X'


# --- provenance ------------------------------------------------------------------


def test_provenance_records_mode_hash_and_enforcement(tmp_path):
    loaded = load("guard", path=_write(tmp_path, FIXTURE_TEXT, name="mine.txt"))
    assert provenance(loaded) == {
        "name": "guard",
        "source": paper_prompts.SOURCE_CUSTOM,
        "filename": "mine.txt",
        "mode": MODE_CUSTOM,
        "sha256": sha256(normalize(FIXTURE_TEXT)),
        "hash_enforced": False,
    }


def test_provenance_records_enforcement_when_reproduction_ran(tmp_path, synthetic_spec):
    loaded = load("fixture", path=_write(tmp_path, FIXTURE_TEXT), mode=MODE_REPRODUCTION)
    p = provenance(loaded)
    assert p["mode"] == MODE_REPRODUCTION and p["hash_enforced"] is True


def test_provenance_carries_no_directory_path(tmp_path):
    """Provenance travels with published artifacts; it must not embed local paths."""
    nested = tmp_path / "some" / "private" / "dir"
    nested.mkdir(parents=True)
    loaded = load("guard", path=_write(nested, FIXTURE_TEXT, name="mine.txt"))
    assert provenance(loaded)["filename"] == "mine.txt"
    assert str(tmp_path) not in repr(provenance(loaded))


# --- the notice is a CLI concern, gated on reproduction mode ---------------------


def test_loader_prints_nothing_in_either_mode(tmp_path, capsys, synthetic_spec):
    load("fixture", path=_write(tmp_path, FIXTURE_TEXT), mode=MODE_CUSTOM)
    load("fixture", path=_write(tmp_path, FIXTURE_TEXT), mode=MODE_REPRODUCTION)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_notice_emitted_only_in_reproduction_mode(capsys):
    assert paper_prompts.emit_reproduction_notice(False) is False
    assert capsys.readouterr().err == ""

    assert paper_prompts.emit_reproduction_notice(True) is True
    err = capsys.readouterr().err
    assert "CC BY-NC-SA" in err and paper_prompts.PAPER_URL in err


def test_notice_goes_to_stderr_not_stdout(capsys):
    paper_prompts.emit_reproduction_notice(True)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err != ""


# --- the recorded specs stay well-formed -----------------------------------------


def test_specs_are_well_formed():
    assert set(PROMPT_SPECS) == {"guard", "judge"}
    for name, spec in PROMPT_SPECS.items():
        assert spec.name == name
        assert spec.placeholder.startswith("{") and spec.placeholder.endswith("}")
        assert len(spec.sha256) == 64 and int(spec.sha256, 16) >= 0
        assert spec.figure and spec.role


def test_mode_for_maps_the_reproduction_flag():
    assert paper_prompts.mode_for(True) == MODE_REPRODUCTION
    assert paper_prompts.mode_for(False) == MODE_CUSTOM
