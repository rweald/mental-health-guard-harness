# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Guards for the Modal fine-tune sweep — parsed via ast / regenerated from the lock, so
the tests need no ``modal`` install:

- SEEDS/JUDGES are duplicated as literals (the shared config isn't importable in the
  modal-launch env); guard they can't drift from the real config.
- the image installs from a lock-exported requirements file; guard it stays current."""

import ast
import subprocess
from pathlib import Path

from mh_guard_harness.training.config import SEEDS
from mh_guard_harness.training.train import adapter_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
MODAL_SCRIPT = REPO_ROOT / "scripts" / "train" / "modal_finetune.py"
REQUIREMENTS = REPO_ROOT / "scripts" / "train" / "modal-requirements.txt"

# Keep in sync with the uv export command documented in modal_finetune.py's image comment.
_EXPORT_CMD = [
    "uv", "export", "--no-emit-project", "--no-dev",
    "--no-emit-package", "torch", "--no-hashes", "--frozen",
]


def _module_literal(name):
    for node in ast.parse(MODAL_SCRIPT.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found as a module-level assignment in modal_finetune.py")


def _req_lines(text):
    """Requirement lines only — drop comments (incl. the generated header) and blank lines."""
    return sorted(
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    )


def test_modal_seeds_match_config():
    # drift here would train the cloud sweep on different seeds than everything else
    assert tuple(_module_literal("SEEDS")) == tuple(SEEDS)


def test_modal_judges_are_the_three_label_sets():
    assert _module_literal("JUDGES") == ["glm-4.6", "glm-5.2", "deepseek-v4-flash"]


def test_modal_adapter_name_matches_train_layout():
    """--skip-existing matches volume entries against this format; if it drifted from
    adapter_dir, every cell would look unfinished and a resumed sweep would re-spend."""
    fmt = _module_literal("ADAPTER_NAME_FMT")
    assert fmt.format(judge="glm-5.2", seed=1) == Path(adapter_dir("glm-5.2", 1)).name


def test_modal_requirements_current_with_lock():
    """The committed requirements must be regenerable from uv.lock — guards it going stale
    when deps change (the drift the lock-export approach is meant to prevent)."""
    fresh = subprocess.run(
        _EXPORT_CMD, cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    assert _req_lines(fresh) == _req_lines(REQUIREMENTS.read_text()), (
        "scripts/train/modal-requirements.txt is stale — regenerate with the uv export "
        "command documented in scripts/train/modal_finetune.py"
    )
