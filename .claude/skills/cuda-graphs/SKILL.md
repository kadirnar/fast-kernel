---
name: cuda-graphs
description: How to capture a launch-bound workload (whole forward, encode/decode, a decode step, an RVQ chain) into CUDA graphs with static buffers and shape buckets using fastkernel.backends.graphs.Graphed. Use when the profile says GPU busy < 60 % or hundreds of small launches.
argument-hint: [callable-to-capture]
allowed-tools: Read, Edit, Write, Glob, Grep
---

# CUDA graphs

`from fastkernel.backends.graphs import Graphed, ShapeBucketedGraphs`

```python
g = Graphed(fn, example_args, example_kwargs)   # warms up on a side stream, captures once
out = g(*args, **kwargs)                        # copies into static inputs, replays, returns static outputs
bucketed = ShapeBucketedGraphs(fn)              # one graph per (shape, dtype) signature, lazy capture
```

## Requirements
- Static shapes: pad/bucket inputs (audio length → frame multiples; tokens → static KV cache with
  `cache_implementation="static"` or a hand-rolled cache; images → fixed size).
- No CPU syncs inside `fn` (`.item()`, `.cpu()`, `print`, data-dependent Python branches); no allocations
  that change between calls (pre-allocate buffers); RNG state handled by torch's graph-safe generators.
- Outputs are the graph's static buffers: clone before feeding them into the next call of another graph
  (encode → decode round trip) or when the harness keeps them.
- transformers outputs are ModelOutput dataclasses: capture the tensor-returning inner function and rebuild
  the same output type around the static tensors so the public API is unchanged.
- Capture in `apply(model, ctx)` (warm-up + capture are excluded from timing); keep a per-shape dict.

## Verify
Determinism gate must pass (replays are deterministic); compare kernel counts before/after in the profile
(one `graph launch` replaces N launches); check peak VRAM (static buffers are extra memory).
