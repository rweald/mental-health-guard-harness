# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""World-accessible dir/file creation for cross-boundary (root-in-docker vs local)
result artifacts."""

import os
import stat

from mh_guard_harness.io_utils import ensure_shared_dir, open_shared


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_ensure_shared_dir_creates_nested_0777(tmp_path):
    target = tmp_path / "results" / "eval" / "judge_labels"
    ensure_shared_dir(target)
    assert target.is_dir()
    # every newly created component is world-accessible
    for p in (tmp_path / "results", tmp_path / "results" / "eval", target):
        assert _mode(p) == 0o777


def test_open_shared_makes_file_0666_and_creates_parents(tmp_path):
    f = tmp_path / "a" / "b" / "scores.jsonl"
    with open_shared(f) as fh:
        fh.write("hi")
    assert f.read_text() == "hi"
    assert _mode(f) == 0o666
    assert _mode(tmp_path / "a") == 0o777


def test_ensure_shared_dir_idempotent_on_existing(tmp_path):
    ensure_shared_dir(tmp_path / "x")
    # second call must not raise even though the dir already exists
    ensure_shared_dir(tmp_path / "x")
    assert (tmp_path / "x").is_dir()
