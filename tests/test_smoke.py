# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the env scaffold.

These verify the environment is wired up and the torch-free design rule holds,
before any real code exists.
"""

import subprocess
import sys


def test_package_imports_and_has_version():
    import mh_guard_harness as mje

    assert mje.__version__


def test_top_level_import_is_torch_free():
    """Importing the top-level package must not drag in torch.

    Run in a fresh subprocess so another test can't have imported torch first.
    This is the automated guard on the modeling-quarantine design rule.
    """
    code = (
        "import mh_guard_harness, sys; "
        "leaked = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.')); "
        "assert not leaked, leaked"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
