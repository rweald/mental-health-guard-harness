# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Filesystem helpers for eval artifacts.

Result files are regenerable and not precious, but they get written from two sides of
a boundary: root inside the GPU/eval container and the local (non-root) user. Whoever
creates a results directory first should leave it world-accessible, so the other side
never hits a permission lockout. These helpers make created dirs 0777 and files 0666.
"""

from __future__ import annotations

import os
from pathlib import Path

DIR_MODE = 0o777
FILE_MODE = 0o666


def _chmod(path: Path, mode: int) -> None:
    # A dir/file created by another user on a prior run may not be ours to chmod; the
    # content still gets written, so a failed chmod is non-fatal.
    try:
        os.chmod(path, mode)
    except PermissionError:
        pass


def ensure_shared_dir(path) -> Path:
    """``mkdir -p`` with each newly created component chmod'd world-accessible."""
    p = Path(path)
    to_make = []
    cur = p
    while not cur.exists():
        to_make.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    for d in reversed(to_make):
        d.mkdir(exist_ok=True)
        _chmod(d, DIR_MODE)
    return p


def open_shared(path, mode: str = "w"):
    """Open a file for writing under a world-accessible dir tree, and make the file
    itself world-readable/writable so root-in-docker output stays usable locally."""
    p = Path(path)
    ensure_shared_dir(p.parent)
    f = open(p, mode)
    _chmod(p, FILE_MODE)
    return f
