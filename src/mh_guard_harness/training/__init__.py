# Copyright 2026 Ryan Weald
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune (LoRA) training lineage.

Turns the judge-labeled synthetic dialogues into supervised examples and trains one
Qwen3Guard LoRA per label set. Training-lineage only: no testset data crosses here.
"""
