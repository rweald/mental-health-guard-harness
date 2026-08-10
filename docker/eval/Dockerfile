# GPU runtime for the model side of the harness (eval scoring; also the
# LoRA fine-tune). Built on the NGC PyTorch image whose torch/CUDA stack is validated
# against this box (GB10 / sm_121 / Blackwell). NGC release 25.09 is the ceiling that
# runs here — do not bump without re-verifying on the hardware.
FROM nvcr.io/nvidia/pytorch:25.09-py3

# uv for fast installs (project standard). NGC ships pip; we add uv on top.
RUN pip install --no-cache-dir uv

# Project model-side deps installed ON TOP OF NGC's prebuilt CUDA torch.
#   - torch is intentionally NOT installed here — it comes from the base image.
#     (uv/pip see NGC torch already satisfies the requirement and leave it alone.)
#   - This mirrors pyproject's `modeling` group plus `datasets` for the testset.
#   - trl/peft drive the LoRA fine-tune: peft is the LoRA implementation,
#     trl's SFTTrainer the supervised-training wrapper. Neither compiles native
#     code (unlike bitsandbytes/FA2) — they're pure-Python over NGC's torch — so
#     bf16 LoRA adds no compile-toolchain risk on Blackwell. Pinned to versions
#     verified against this image's transformers 5.x: both leave transformers and
#     torch untouched (confirmed by a resolver dry-run). Re-verify these pins if
#     the NGC base (and thus transformers) is bumped.
#   - Metrics/stats (numpy/scipy/scikit-learn/statsmodels/pandas) deliberately live
#     in the LOCAL torch-free env, not here: it keeps the image lean and avoids a
#     numpy ABI bump breaking NGC's compiled torch. The container emits per-turn
#     scores; the local env computes AUROC / FPR@TPR from them.
# --break-system-packages: the NGC base is an "externally managed" env (PEP 668);
# installing alongside its system torch is exactly what we intend here.
RUN uv pip install --system --break-system-packages --no-cache \
      transformers \
      accelerate \
      datasets \
      "peft==0.20.0" \
      "trl==1.9.2"

# The NGC base ships torchao 0.13 (a git preview). peft 0.20's LoRA dispatcher eagerly
# version-checks torchao and RAISES on anything below 0.16 — even though we do plain
# bf16 LoRA and never touch torchao. Remove it so the dispatcher cleanly skips its
# torchao handler. We use no quantization, so nothing else in our path needs it.
RUN uv pip uninstall --system --break-system-packages torchao

WORKDIR /work
ENV PYTHONPATH=/work/src

# Default command doubles as a readiness check (and a trl/peft import smoke).
CMD ["python", "-c", "import torch, transformers, peft, trl; print('eval container ready — torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| peft', peft.__version__, '| trl', trl.__version__)"]
