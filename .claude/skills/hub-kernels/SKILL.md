---
name: hub-kernels
description: How to use pre-built kernels from the Hugging Face `kernels` hub (flash-attn, activations, norms, quant) in fast-kernel candidates when no compiler is available or to get a strong baseline quickly.
argument-hint: [kernel-name]
allowed-tools: Read, Edit, Write, Glob, Grep, Bash(uv pip *)
---

```python
from kernels import get_kernel
act = get_kernel("kernels-community/activation")      # e.g. act.silu_and_mul(out, x)
fa = get_kernel("kernels-community/flash-attn2")      # already in the local HF cache on this machine
```
- `uv pip install kernels` if missing; kernels are fetched once into the HF cache and matched to the
  torch/CUDA build (check `fast-kernel probe` → hub-kernels → cached_kernels).
- Use them as drop-in replacements inside `apply(model, ctx)` (module swap or forward patch), then measure:
  a hub kernel that is not faster than the hand-written CUDA kernel it replaces is not an improvement.
- Attention: `attn_implementation="kernels-community/flash-attn2"` works for Transformers models that
  support the kernels integration; otherwise call the kernel from a patched attention forward.
