---
model: lfm25
objective: "Maximise LFM2.5-1.2B-Instruct greedy decode throughput (batch 1, static prompt) and prefill speed on this GPU while keeping greedy tokens and top-1 logits identical to the bf16 Transformers reference."
target_metric: latency_ms
direction: minimize
min_improvement: 0.01
continuous: true
primary_workload: decode
gates:
  precision: strict
  determinism: exact
  stages: [smoke, shapes, numerical, determinism, edge]
bench:
  warmup: 3
  repeats: 15
  ramp_seconds: 1.0
  timeout_seconds: 2400
  profile_every_experiment: true
model_args:
  variant: 1.2b-instruct
  dtype: bfloat16
  prefill_tokens: 512
  decode_tokens: 64
  decode_prompt_tokens: 64
  batch: 1
protected: [GOAL.md, spec.py, harness/**, .fast-kernel/**, experiments/**, results.tsv]
---

# Goal

Speed up `LiquidAI/LFM2.5-1.2B-Instruct` (transformers `Lfm2ForCausalLM`, bf16) for the two workloads
that matter on a single GPU: `decode` (primary; 64 greedy tokens after a 64-token prompt, latency of the
whole generation) and `prefill` (512 tokens). Keep `AutoModelForCausalLM` semantics: `generate()` and
`forward()` must keep working with the same inputs.

# Policy (the human decides; the agent never changes it)

- strict: generated tokens identical; prefill top-1 identical on >= 99.5 % positions, top-5 overlap >= 0.9.
- tolerant: >= 90 % identical greedy tokens (bf16 reordering is allowed to flip near-ties).
- `model_args.variant: 350m` gives a faster development loop with the same architecture.


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
