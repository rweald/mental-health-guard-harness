# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Decision-model scoring: question spec, request state, and answer -> P(unsafe).

Decision models answer typed questions about a ``state`` with a probability per allowed
answer, so they need neither the guard prompt template nor logits. Each testset row is
sent once with two questions, and each question produces its own scores file in the
usual seam-contract JSONL, so metrics and comparison code run on either unchanged:

  - ``risk_category`` — a three-way ``choice`` (safe / self-harm / harm to others).
    Score: ``1 - P(safe)``, after normalizing the three probabilities.
  - ``unsafe`` — a yes/no ``noul``. Score: the returned probability that it holds.

Both are first-class results. Neither is picked over the other after the fact.

What the questions *say* is not in this repository — like the guard and judge prompts,
it is loaded from a user-supplied JSON file (the "decision spec"). This module fixes
only the structure: question names, types, and option keys, which the scorer depends on.
The spec's hash goes into provenance, and the scorer refuses to resume a run under a
different spec, so every model in a comparison is scored with the identical questions.

The state keeps the target explicit by structure rather than by instruction: prior
turns go in ``conversation_history`` and the user turn being judged is
``message_to_assess``. Context is past-only, exactly as each testset row provides it.

Torch-free, so the scorer runs locally rather than in the GPU container.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPEC_VERSION = 1

# Question names and their output files. Names are part of the request, so changing one
# changes what every model is asked; the file suffixes are what comparison code globs.
CHOICE3 = "risk_category"
YESNO = "unsafe"
SCORE_FILES = {CHOICE3: "choice3", YESNO: "yesno"}

# Option keys for the three-way question. These match metrics.risk_category's subgroup
# names, so a per-category analysis can line predictions up with gold without a mapping.
SAFE = "safe"
CHOICE3_OPTIONS = (SAFE, "self_harm", "harm_to_others")
YESNO_CRITERIA = ("true", "false")

_EXPECTED_TYPES = {CHOICE3: "choice", YESNO: "noul"}
_EXPECTED_CRITERIA = {CHOICE3: set(CHOICE3_OPTIONS), YESNO: set(YESNO_CRITERIA)}

# TypeSafe documents that Jev switches to a slower, less accurate long-context mode for
# text state over about this many bytes. Recorded per row so a model's results can be
# read in light of how many rows crossed it; the threshold itself is Jev's, not ours.
LONG_STATE_BYTES = 32_000

# Failure kinds: the model declined (``refusal``), answered in a shape we can't score
# (``invalid``), or the request itself failed (``error``). None are dropped silently.
REFUSAL = "refusal"
INVALID = "invalid"
ERROR = "error"


class SpecError(ValueError):
    """The decision spec file is missing, unreadable, or structurally wrong. Messages
    name the offending field but never quote spec text."""


@dataclass(frozen=True)
class LoadedSpec:
    questions: dict[str, Any]
    sha256: str
    filename: str


def _canonical(obj: Any) -> str:
    """Stable serialization for hashing: key order and whitespace don't change the hash,
    any change to question text does."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def load_spec(path) -> LoadedSpec:
    """Read and validate the decision spec at ``path``. There is no default path."""
    if path is None:
        raise SpecError("no decision spec path configured (CLI --spec)")
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise SpecError(f"cannot read decision spec at {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise SpecError(f"decision spec at {path} is not valid JSON: {e}") from e

    if not isinstance(data, dict) or data.get("spec_version") != SPEC_VERSION:
        raise SpecError(f"decision spec at {path}: spec_version must be {SPEC_VERSION}")
    questions = data.get("questions")
    if not isinstance(questions, dict) or set(questions) != set(_EXPECTED_TYPES):
        got = sorted(questions) if isinstance(questions, dict) else type(questions).__name__
        raise SpecError(
            f"decision spec at {path}: questions must be exactly "
            f"{sorted(_EXPECTED_TYPES)}, got {got}"
        )
    for name, q in questions.items():
        where = f"decision spec at {path}, question {name!r}"
        if not isinstance(q, dict) or set(q) != {"type", "instructions", "criteria"}:
            raise SpecError(f"{where}: must have exactly type, instructions, criteria")
        if q["type"] != _EXPECTED_TYPES[name]:
            raise SpecError(f"{where}: type must be {_EXPECTED_TYPES[name]!r}")
        if not _nonempty_str(q["instructions"]):
            raise SpecError(f"{where}: instructions must be a non-empty string")
        crit = q["criteria"]
        if not isinstance(crit, dict) or set(crit) != _EXPECTED_CRITERIA[name]:
            raise SpecError(
                f"{where}: criteria keys must be exactly {sorted(_EXPECTED_CRITERIA[name])}"
            )
        bad = sorted(k for k, v in crit.items() if not _nonempty_str(v))
        if bad:
            raise SpecError(f"{where}: criteria {bad} must be non-empty strings")

    return LoadedSpec(
        questions=questions,
        sha256=hashlib.sha256(_canonical(questions).encode("utf-8")).hexdigest(),
        filename=path.name,
    )


def spec_provenance(spec: LoadedSpec) -> dict:
    """Basename and hash only — provenance travels with results and shouldn't carry a
    home directory, or the spec's text."""
    return {"filename": spec.filename, "sha256": spec.sha256, "spec_version": SPEC_VERSION}


def build_state(messages: list[dict]) -> dict:
    """The request state for one testset row: prior turns plus the user turn to judge.

    The row's final message must be the user turn being classified — that is the
    testset's shape, and judging anything else would silently change the task."""
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("a row's final message must be the user turn being assessed")
    return {
        "conversation_history": [
            {"role": m["role"], "content": m["content"]} for m in messages[:-1]
        ],
        "message_to_assess": messages[-1]["content"],
    }


def state_bytes(state: dict) -> int:
    return len(_canonical(state).encode("utf-8"))


@dataclass(frozen=True)
class Scored:
    """One question's answer turned into a score, or the reason it couldn't be."""

    p_unsafe: float | None = None
    extra: dict | None = None
    failure: str | None = None
    detail: str | None = None


def _finite_prob(x: Any) -> float | None:
    """A JSON number in [0, 1], else ``None``. Booleans and numeric strings are not
    probabilities: ``true`` would otherwise score as 1.0 and quietly turn an arm's
    scores into hard 0/1 labels."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    v = float(x)
    return v if math.isfinite(v) and 0.0 <= v <= 1.0 else None


def _check_answer(answers: dict, name: str) -> tuple[dict | None, Scored | None]:
    a = answers.get(name) if isinstance(answers, dict) else None
    if not isinstance(a, dict):
        return None, Scored(failure=INVALID, detail="no answer for this question")
    if a.get("type") == REFUSAL:
        return None, Scored(failure=REFUSAL, detail=str(a.get(REFUSAL) or "")[:500])
    if a.get("type") != _EXPECTED_TYPES[name]:
        return None, Scored(failure=INVALID, detail=f"answer type {a.get('type')!r}")
    return a, None


def score_choice3(answers: dict) -> Scored:
    """``1 - P(safe)`` from the three-way answer, with the probabilities renormalized
    over the three options (a provider that returns them unnormalized shouldn't shift
    the score). Missing or out-of-range probabilities are invalid, not clamped."""
    a, fail = _check_answer(answers, CHOICE3)
    if fail:
        return fail
    probs = a.get("probabilities")
    if not isinstance(probs, dict):
        return Scored(failure=INVALID, detail=f"probabilities is {type(probs).__name__}")
    if set(probs) != set(CHOICE3_OPTIONS):
        return Scored(failure=INVALID, detail=f"probability keys {sorted(map(str, probs))}")
    p = {k: _finite_prob(v) for k, v in probs.items()}
    total = sum(v for v in p.values() if v is not None)
    if any(v is None for v in p.values()) or total <= 0:
        return Scored(failure=INVALID, detail="probabilities not finite in [0, 1]")
    p = {k: v / total for k, v in p.items()}
    return Scored(
        p_unsafe=1.0 - p[SAFE],
        extra={
            "probabilities": p,
            "choice": a.get("choice"),
            "confidence": a.get("confidence"),
        },
    )


def score_yesno(answers: dict) -> Scored:
    """The yes/no answer's probability that the message is unsafe."""
    a, fail = _check_answer(answers, YESNO)
    if fail:
        return fail
    p = _finite_prob(a.get("noul"))
    if p is None:
        return Scored(failure=INVALID, detail="probability not finite in [0, 1]")
    return Scored(p_unsafe=p)


SCORERS = {CHOICE3: score_choice3, YESNO: score_yesno}


# --- Running and deriving ------------------------------------------------------------
#
# A run has two halves. ``run`` makes the paid calls and appends every raw response to
# a cache file, one line per row; it resumes by skipping rows already answered, so a
# crash or a 429 storm costs only the rows that didn't land. ``derive`` turns the cache
# into the scores files with no network at all, so a scoring-rule change or a bug fix
# never means paying for the calls again.


class CacheMismatchError(RuntimeError):
    """The cache was written under a different spec or model than this run's."""


class CacheCorruptError(RuntimeError):
    """A cache line other than the last one doesn't parse — not a crash mid-write."""


def row_sha256(row: dict) -> str:
    """Identity of a row's request: the hash of the state it sends. Cache entries are
    matched to rows by this, never by index, so answers cached for one rows file (a
    smoke set, say) can't be paired with a different file's rows that share an index."""
    return hashlib.sha256(_canonical(build_state(row["prompt"])).encode("utf-8")).hexdigest()


def read_cache(path, *, spec_sha256: str, model_id: str) -> dict[str, dict]:
    """Latest cache entry per row hash. Refuses a cache written under another spec or
    model: mixing them would put two different questions into one scores file.

    A final line that doesn't parse is what a crash mid-write leaves behind, so it is
    skipped with a warning and that row is simply pending again. A bad line anywhere
    else is real corruption and raises."""
    import warnings

    path = Path(path)
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            if i == len(lines) - 1:
                warnings.warn(f"{path}: skipping a truncated final line; that row will "
                              "be requested again", stacklevel=2)
                break
            raise CacheCorruptError(f"{path}: line {i + 1} is not valid JSON: {e}") from e
        if entry.get("spec_sha256") != spec_sha256 or entry.get("model_id") != model_id:
            raise CacheMismatchError(
                f"{path} was written for model {entry.get('model_id')!r} under spec "
                f"{str(entry.get('spec_sha256'))[:12]}; this run is {model_id!r} under "
                f"{spec_sha256[:12]}. Use a fresh --outdir rather than mixing them."
            )
        if "row_sha256" not in entry:
            raise CacheMismatchError(
                f"{path} predates row-hash matching, so its entries can't be safely "
                "paired with rows. Use a fresh --outdir."
            )
        latest[entry["row_sha256"]] = entry
    return latest


def pending_rows(rows: list[dict], cache: dict[str, dict]) -> list[dict]:
    """Rows with no successful response yet. Failed requests are retried on the next
    run; answered rows (refusals included — a refusal is an answer) are not."""
    return [r for r in rows if not cache.get(row_sha256(r), {}).get("ok")]


def check_rows(rows: list[dict]) -> None:
    """Fail before any paid call if a row can't be turned into a request, or if two
    rows share an index (the scores files would then be ambiguous)."""
    seen = set()
    for row in rows:
        index = row.get("index")
        if index in seen:
            raise ValueError(f"duplicate row index {index!r}")
        seen.add(index)
        try:
            build_state(row["prompt"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"row {index!r} can't be sent: {e}") from e


def run(
    model,
    rows: list[dict],
    spec: LoadedSpec,
    cache_path,
    *,
    model_id: str,
    workers: int = 8,
    progress=None,
) -> int:
    """Request a decision for every pending row and append each outcome to the cache.
    Returns how many requests completed.

    Stops sending on Ctrl-C or on a fatal API error (bad key, no credits): queued rows
    are cancelled, the few already in flight finish and are cached, and the exception
    propagates. Fatal errors are not cached — they say nothing about the row."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .io_utils import open_shared
    from .models import DecisionAPIError

    check_rows(rows)
    cache_path = Path(cache_path)
    cache = read_cache(cache_path, spec_sha256=spec.sha256, model_id=model_id)
    todo = pending_rows(rows, cache)
    lock = threading.Lock()
    done = 0

    def _one(row: dict) -> None:
        nonlocal done
        state = build_state(row["prompt"])
        entry = {
            "index": row["index"],
            "row_sha256": row_sha256(row),
            "model_id": model_id,
            "spec_sha256": spec.sha256,
            "state_bytes": state_bytes(state),
        }
        t0 = time.monotonic()
        try:
            decision = model.decide(state, spec.questions)
            entry.update(ok=True, response=decision.raw)
        except DecisionAPIError as e:
            if e.fatal:
                raise
            entry.update(ok=False, error={"status": e.status, "message": str(e)[:1000]})
        entry["latency_s"] = round(time.monotonic() - t0, 4)
        line = json.dumps(entry) + "\n"
        with lock:
            f.write(line)
            f.flush()
            done += 1
            if progress:
                progress(done, len(todo), entry)

    with open_shared(cache_path, "a") as f:
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            for fut in as_completed([ex.submit(_one, row) for row in todo]):
                fut.result()
        except BaseException:
            ex.shutdown(wait=True, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
    return done


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def _is_positive(label: str) -> bool:
    return label != SAFE


def derive(
    rows: list[dict],
    cache_path,
    outdir,
    *,
    key: str,
    spec: LoadedSpec,
    model_id: str,
    provenance_base: dict,
) -> dict:
    """Write one scores file per question from the cache, plus a failures file and a
    provenance sidecar per scores file.

    Rows that couldn't be scored are left out of the scores file and listed in the
    failures file. That makes the scores file the "failures excluded" view; the
    comparison report adds the "failures scored as unsafe" view from the failures file.
    Excluding rows changes the set an arm is scored on — an arm that refuses the most
    explicit turns gets an easier test — so provenance counts failures by gold class,
    and the summary's ``warnings`` says so whenever any row was left out."""
    from collections import Counter

    from .io_utils import open_shared

    check_rows(rows)
    cache = read_cache(cache_path, spec_sha256=spec.sha256, model_id=model_id)
    outdir = Path(outdir)
    entries = {r["index"]: cache.get(row_sha256(r)) for r in rows}
    answered = [e for e in entries.values() if e and e.get("ok")]
    resolved = Counter(e["response"].get("model") for e in answered)
    providers = Counter(e["response"].get("provider") for e in answered)
    costs = [(e["response"].get("usage") or {}).get("cost") for e in answered]
    latencies = [e["latency_s"] for e in answered]
    run_stats = {
        "rows": len(rows),
        "answered": len(answered),
        "unanswered": len(rows) - len(answered),
        "resolved_models": dict(resolved),
        "providers": dict(providers),
        "cost_usd": round(sum(c for c in costs if isinstance(c, (int, float))), 6),
        "latency_s": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
        "rows_over_long_state_bytes": sum(
            1 for e in answered if e["state_bytes"] > LONG_STATE_BYTES
        ),
    }

    summary = {}
    warnings = []
    failures_path = outdir / f"{key}.failures.jsonl"
    with open_shared(failures_path) as ff:
        for question, suffix in SCORE_FILES.items():
            scores_path = outdir / f"{key}__{suffix}_scores.jsonl"
            kinds: Counter = Counter()
            by_class = {"positive": Counter(), "safe": Counter()}
            n_scored = 0
            with open_shared(scores_path) as sf:
                for row in rows:
                    entry = entries[row["index"]]
                    if entry is None or not entry.get("ok"):
                        scored = Scored(
                            failure=ERROR,
                            detail=(entry or {}).get("error", {}).get("message", "not run"),
                        )
                    else:
                        scored = SCORERS[question](entry["response"].get("answers"))
                    if scored.failure:
                        kinds[scored.failure] += 1
                        cls = "positive" if _is_positive(row["label"]) else "safe"
                        by_class[cls][scored.failure] += 1
                        ff.write(json.dumps({
                            "index": row["index"], "question": question,
                            "kind": scored.failure, "detail": scored.detail,
                            "label": row["label"],
                        }) + "\n")
                        continue
                    record = {
                        "index": row["index"],
                        "p_unsafe": scored.p_unsafe,
                        "label": row["label"],
                        "annotator_labels": row["annotator_labels"],
                    }
                    if scored.extra:
                        record.update(scored.extra)
                    sf.write(json.dumps(record) + "\n")
                    n_scored += 1
            failures_by_class = {c: dict(k) for c, k in by_class.items()}
            prov = {
                **provenance_base,
                "question": question,
                "scored": n_scored,
                "failures": dict(kinds),
                "failures_by_class": failures_by_class,
                "run": run_stats,
                "decision_spec": spec_provenance(spec),
            }
            scores_path.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=2))
            summary[question] = {"path": str(scores_path), "scored": n_scored, **dict(kinds)}
            if kinds:
                warnings.append(
                    f"{question}: {sum(kinds.values())} of {len(rows)} rows left out of "
                    f"{scores_path.name} (positives {dict(by_class['positive'])}, safe "
                    f"{dict(by_class['safe'])}) — metrics on this file exclude them"
                )
    return {"questions": summary, "run": run_stats, "warnings": warnings}
