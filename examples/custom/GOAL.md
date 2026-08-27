---
model: custom
objective: "Make the forward pass of my torch module as fast as possible without changing its outputs."
target_metric: latency_ms
direction: minimize
min_improvement: 0.01
continuous: true
gates:
  precision: strict
  determinism: exact
  stages: [smoke, shapes, numerical, determinism, edge]
bench:
  warmup: 5
  repeats: 50
  ramp_seconds: 1.0
  timeout_seconds: 900
  profile_every_experiment: true
model_args:
  loader: "spec:build_model"
  input_shapes: [[1, 3, 224, 224]]
  input_dtype: float32
  batch_sweep: [4]
protected: [GOAL.md, spec.py, harness/**, .fast-kernel/**, experiments/**, results.tsv]
---

# Goal

Edit `spec.py` (`build_model`, and optionally the `Spec` class for custom workloads / comparisons), then
`fast-kernel baseline`. Everything else is generic: gates compare candidate outputs to the fp32 reference
with allclose; benchmarks use the fixed protocol; profiling ranks hotspots by measured headroom (share × (1 − roofline efficiency)).

# Quality contract

Faster is only accepted without a loss of quality. The default policy `gates.precision: strict` means
the outputs must match the original model (identical discrete outputs, floating-point outputs within
the spec tolerance), deterministically, on the edge workloads too. Only a human changes this file; the
agent never loosens gates, skips stages or shrinks workloads.

# How to decide what to optimize

Nothing is prescribed. Measure first: `fast-kernel baseline` and `fast-kernel profile` rank the targets of
*this* model on *this* machine in PLAN.md; `capabilities.json` says which backends compile here. Every
hypothesis comes from those measurements and from what earlier experiments taught (KNOWLEDGE.md), never
from assumptions about the hardware. Any technique and any backend may be tried; only the quality
contract limits what is kept.
