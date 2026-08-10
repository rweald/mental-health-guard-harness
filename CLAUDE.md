# mental-health-guard-harness

A harness for training and evaluating turn-level safety classifiers on multi-turn mental
health conversations. See @README.md for what it is and @docs/adapting-this-harness.md for
the seams.

## Prompts are not in this repo

Both prompts (`guard`, `judge`) load from user-supplied files at call time — see
@prompts/README.md. Never add prompt text to a source file, test fixture, docstring, or
error message; if a test needs one, write a short synthetic fixture with the placeholder.

## Running costly or contended work — get sign-off first

Two tiers of things the assistant must not kick off on its own:

1. **Paid LLM-vendor calls** (OpenRouter, Anthropic, OpenAI — judge labeling, conversation
   generation, any paid API call): never run these. Prepare the exact command and hand it to
   the user to run manually.
2. **Local GPU inference** (e.g. the docker eval container loading a model onto the GPU —
   scoring an eval set, fine-tuning): the assistant may run it, but must **explicitly ask for
   permission first** every time, not start it unprompted.

Everything else — pytest, ruff, data probes, building images, non-GPU container work — is
fine to run directly without asking.
