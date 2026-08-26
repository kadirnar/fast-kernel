---
model: mimi
objective: "Make Mimi encode+decode of 1 s of 24 kHz audio (batch 1) as fast as possible while keeping the transformers API, exact discrete codes (strict) and waveform tolerances."
target_metric: latency_ms
direction: minimize
min_improvement: 0.01
continuous: true
primary_workload: roundtrip_1s
gates:
  precision: strict
  determinism: exact
  stages: [smoke, shapes, numerical, determinism, edge]
bench:
  warmup: 5
  repeats: 50
  ramp_seconds: 1.0
  timeout_seconds: 1500
  profile_every_experiment: true
model_args:
  seconds: 1.0
  sweep: [0.25, 5.0]
protected: [GOAL.md, spec.py, harness/**, .fast-kernel/**, experiments/**, results.tsv]
---

# Goal

Optimize `kyutai/mimi` (transformers `MimiModel`) end to end: SEANet encoder, encoder transformer,
32-stage residual vector quantizer, decoder transformer, SEANet decoder. The reference is the stock
fp32 model (TF32 off). Every accepted experiment must keep the public API
(`encode(audio, padding_mask).audio_codes`, `decode(codes, padding_mask).audio_values`) intact.

# Policy

- `precision: strict` = the discrete codes must be identical to the fp32 oracle and decoded audio must be
  allclose (rtol 2e-4, atol 2e-5). Switch to `tolerant` (code match >= 85 %, decode SNR >= 35 dB,
  reconstruction within 0.5 dB) to allow bf16 tensor-core arithmetic in the quantizer path.
- Measured on the primary workload `roundtrip_1s`; sweeps at 0.25 s and 5 s must stay correct.
- Edge gates: 50 ms input, non-multiple-of-frame length, batch 2.

# Where to start (measured priors, see PLAN.md after `fast-kernel profile`)

1. The stock path is launch bound (~1250 kernels for 1 s). CUDA-graph the encode and decode calls
   (`fastkernel.backends.graphs.Graphed`, shape buckets by audio length).
2. Fuse the codebook search (`MimiEuclideanCodebook`): distance + argmin without cdist, all 32 stages in
   one kernel.
3. Fuse the transformer blocks (LN+QKV+RoPE, attention+O+residual, LN+FC1+GELU, FC2).
4. Rewrite SEANet convolutions as implicit GEMMs with fused ELU epilogues.
