# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Turn-level safety classification for multi-turn mental health conversations.

Generate synthetic patient/clinician dialogues, label every patient turn with a
full-context judge, distil those labels into a small turn-level classifier, and
score the result.

Design rule: this top-level package and everything under it EXCEPT
``mh_guard_harness.modeling`` must import without pulling in torch or
transformers. That keeps the generation loop, labeling, metrics, and stats
runnable (and unit-testable) on CPU with no GPU dependency. See the ``modeling``
subpackage docstring.
"""

__version__ = "0.0.1"
