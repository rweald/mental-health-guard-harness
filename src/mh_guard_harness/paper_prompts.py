# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Load the two prompts this harness needs from user-supplied files.

The harness does not ship prompt text. Both the turn-level classifier prompt (the
"guard" prompt) and the full-context labeling prompt (the "judge" prompt) are loaded
at call time from paths the caller provides. This keeps the harness's own license
clean: the reference prompts come from MindGuard (Farinhas et al.,
arXiv:2602.00950), which is licensed CC BY-NC-SA 4.0, so they cannot be vendored
into an Apache-2.0 repository. Supply your own, or transcribe the reference ones
from the paper into local files that are never committed.

Two modes:

  - ``custom`` (default): any prompt is accepted so long as it carries the required
    placeholder exactly once. Its hash is computed and recorded in provenance, never
    compared. This is the mode for your own prompts.
  - ``reproduction``: the prompt must additionally hash-match the reference value
    recorded in ``PROMPT_SPECS``, so a transcription slip fails loudly instead of
    silently changing the method. Use this when reproducing the published results.

Only SHA-256 hashes of the reference prompts are recorded here — never their text.
A hash confirms a correct transcription; it does not reveal one.

The loader is deliberately quiet (no printing, no logging) and does no caching:
the files are tiny, and a cache keyed on anything less than the full path plus mode
is a correctness bug waiting to happen. The licensing notice for reproduction mode
is emitted by callers at the CLI boundary via ``reproduction_notice()``.
"""

from __future__ import annotations

import hashlib
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PAPER_URL = "https://arxiv.org/abs/2602.00950"

MODE_CUSTOM = "custom"
MODE_REPRODUCTION = "reproduction"
MODES = (MODE_CUSTOM, MODE_REPRODUCTION)

# What ``LoadedPrompt.source`` reports: whether the file's content matches the
# recorded reference hash. Informative even in custom mode — a run's provenance
# then shows whether it used the reference prompt or a substitute.
SOURCE_REFERENCE = "reference"
SOURCE_CUSTOM = "custom"


@dataclass(frozen=True)
class PromptSpec:
    """What a given prompt slot requires, and the hash of the reference version.

    ``sha256`` is the normalized hash of the prompt as published in the paper figure
    named by ``figure``. It is the only thing about the reference text stored here.
    """

    name: str
    placeholder: str
    figure: str
    sha256: str
    role: str


PROMPT_SPECS: dict[str, PromptSpec] = {
    "guard": PromptSpec(
        name="guard",
        placeholder="{conversation}",
        figure="Figure 19 (§7.4)",
        sha256="8193f51874c645436520e4834da88e9e80d36158610e1e7e227f4c23c6dce995",
        role="turn-level classifier prompt (training target format and scoring)",
    ),
    "judge": PromptSpec(
        name="judge",
        placeholder="{transcript}",
        figure="Figure 8 (§7.2)",
        sha256="f42c1fbb9918d3836a13e4aa2d676b7a3541c91a39d6b63ba148bb10d41e85d0",
        role="full-context judge labeling prompt",
    ),
}


@dataclass(frozen=True)
class LoadedPrompt:
    """A prompt read from disk, normalized and validated.

    ``sha256`` is over the normalized text. ``hash_enforced`` records whether the
    hash was actually checked against the reference (reproduction mode) or merely
    observed (custom mode) — provenance must not imply verification that never ran.
    """

    name: str
    text: str
    sha256: str
    source: str
    mode: str
    hash_enforced: bool
    filename: str

    @property
    def spec(self) -> PromptSpec:
        return PROMPT_SPECS[self.name]


class PromptError(Exception):
    """Base class for prompt loading failures."""


class PromptPathError(PromptError):
    """No path was configured, or nothing readable was at it."""


class PromptPlaceholderError(PromptError):
    """The prompt does not carry its required placeholder exactly once."""


class PromptHashMismatchError(PromptError):
    """Reproduction mode was requested and the prompt is not the reference text."""


def normalize(text: str) -> str:
    """Canonical form for hashing and use: NFC, LF line endings, no per-line trailing
    whitespace, no trailing newline.

    Transcribing a prompt out of a PDF picks up incidental differences — a stray
    trailing space, CRLF from a Windows editor, a decomposed accent — that change the
    bytes without changing the prompt. Normalizing before hashing means reproduction
    mode checks the prompt, not the transcription's whitespace habits. Idempotent.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).rstrip("\n")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _describe(spec: PromptSpec, path: object) -> str:
    """Error context: which prompt, where it was looked for, and where to get it.
    Never quotes prompt content."""
    return (
        f"{spec.name} prompt ({spec.role}); expected at {path}; the reference version "
        f"is {spec.figure} of MindGuard (Farinhas et al.), {PAPER_URL}"
    )


def load(name: str, *, path, mode: str = MODE_CUSTOM) -> LoadedPrompt:
    """Read, normalize and validate the prompt named ``name`` from ``path``.

    ``path`` is required and explicit — there is no implicit lookup relative to the
    current working directory, so a run cannot silently pick up a different prompt
    because it started somewhere else.
    """
    try:
        spec = PROMPT_SPECS[name]
    except KeyError:
        raise KeyError(
            f"unknown prompt {name!r}; known prompts: {sorted(PROMPT_SPECS)}"
        ) from None
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if path is None:
        raise PromptPathError(
            f"no path configured for the {_describe(spec, '<unset>')}. Set it "
            f"explicitly (config key '{spec.name}_prompt_path' / CLI --{spec.name}-prompt)."
        )

    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise PromptPathError(f"cannot read the {_describe(spec, path)}: {e}") from e

    text = normalize(raw)

    count = text.count(spec.placeholder)
    if count != 1:
        raise PromptPlaceholderError(
            f"the {_describe(spec, path)} must contain the placeholder "
            f"{spec.placeholder} exactly once; found it {count} times"
        )

    digest = sha256(text)
    enforced = mode == MODE_REPRODUCTION
    if enforced and digest != spec.sha256:
        raise PromptHashMismatchError(
            f"reproduction mode: the {_describe(spec, path)} does not match the "
            f"reference text (expected sha256 {spec.sha256}, got {digest}). Re-check "
            f"the transcription, or use custom mode to run with your own prompt."
        )

    return LoadedPrompt(
        name=spec.name,
        text=text,
        sha256=digest,
        source=SOURCE_REFERENCE if digest == spec.sha256 else SOURCE_CUSTOM,
        mode=mode,
        hash_enforced=enforced,
        filename=path.name,
    )


def render(loaded: LoadedPrompt, value: str) -> str:
    """Substitute ``value`` for the prompt's placeholder.

    Uses ``str.replace`` rather than ``str.format`` so that a user-supplied prompt
    containing braces for any other reason is not a crash or an accidental
    substitution. ``load`` has already guaranteed exactly one placeholder.
    """
    return loaded.text.replace(loaded.spec.placeholder, value)


def load_and_render(name: str, value: str, *, path, mode: str = MODE_CUSTOM) -> str:
    """``load`` then ``render`` — the one-shot form the prompt builders use."""
    return render(load(name, path=path, mode=mode), value)


def mode_for(reproduction: bool) -> str:
    """Map the ``reproduction_mode`` boolean used in configs and CLIs to a mode."""
    return MODE_REPRODUCTION if reproduction else MODE_CUSTOM


def provenance(loaded: LoadedPrompt) -> dict:
    """The record written alongside every run that used this prompt.

    Deliberately carries the file's basename, not its path: provenance travels with
    published artifacts and should not embed someone's home directory.
    """
    return {
        "name": loaded.name,
        "source": loaded.source,
        "filename": loaded.filename,
        "mode": loaded.mode,
        "sha256": loaded.sha256,
        "hash_enforced": loaded.hash_enforced,
    }


def reproduction_notice() -> str:
    """Licensing notice for reproduction-mode runs, emitted to stderr by CLIs.

    Reproduction mode means the reference prompts are in play, and those are
    CC BY-NC-SA 4.0 — a different licence from this harness.
    """
    return (
        "NOTICE: reproduction mode uses prompts from MindGuard (Farinhas et al., "
        f"{PAPER_URL}), which is licensed CC BY-NC-SA 4.0. Those prompts and work "
        "derived from them carry that licence — non-commercial, share-alike, with "
        "attribution. This harness's own code is Apache-2.0; the prompt content is not."
    )


def emit_reproduction_notice(reproduction: bool, stream=None) -> bool:
    """Print the licensing notice to stderr iff reproduction mode is on.

    This is the CLI-boundary helper — ``load`` itself stays silent so that library
    callers get no surprise output. Lives here rather than being re-implemented in
    each script so the gating is written (and tested) exactly once. Returns whether
    it printed, which is what makes the gate testable.
    """
    if not reproduction:
        return False
    print(reproduction_notice(), file=stream if stream is not None else sys.stderr)
    return True
