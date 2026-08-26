---
name: torch-compile
description: How to use torch.compile (inductor) as an optimization step in fast-kernel candidates - modes, dynamic shapes, graph breaks, reduce-overhead (CUDA graphs), partial compilation of hot submodules. Use when fusing elementwise/norm chains cheaply or wrapping a static decode step.
argument-hint: [module-or-function]
allowed-tools: Read, Edit, Write, Glob, Grep, Bash(TORCH_LOGS=* *)
---

- `torch.compile(fn_or_module, mode="max-autotune-no-cudagraphs" | "reduce-overhead" | "max-autotune", dynamic=False, fullgraph=False)`
- Compile only the hot submodules when the whole model graph-breaks (`model.model.layers[i].feed_forward = torch.compile(...)`).
- Find graph breaks: `TORCH_LOGS=graph_breaks uv run python ...`; remove `.item()`, Python-side caches,
  data-dependent control flow from the hot path.
- Shapes: `dynamic=False` + shape buckets, or `torch._dynamo.mark_dynamic` for one dim; every new shape is a
  recompile (excluded from timing only if it happens in warm-up — warm every shape in `apply()`).
- `reduce-overhead` = CUDA graphs managed by inductor; outputs are reused buffers (clone if kept).
- For transformers decode: `model.generation_config.cache_implementation = "static"` then compile
  `model.forward` (see the LFM2.5 campaign GOAL.md).
- Keep TF32 off unless the gate policy allows (`torch.backends.cuda.matmul.allow_tf32`).
- Measure kernel counts before/after in the profile; inductor's Triton kernels appear as `triton_*` names.
