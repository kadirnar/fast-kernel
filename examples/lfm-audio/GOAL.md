---
model: lfm-audio
objective: "Minimise LFM2.5-Audio-1.5B text-to-speech generation latency (backbone + depthformer + Mimi decode) and ASR latency while keeping greedy text/audio tokens identical to the liquid_audio reference."
target_metric: latency_ms
direction: minimize
min_improvement: 0.01
continuous: true
primary_workload: tts
gates:
  precision: strict
  determinism: exact
  stages: [smoke, shapes, numerical, determinism]
bench:
  warmup: 2
  repeats: 7
  ramp_seconds: 0.5
  timeout_seconds: 3600
  profile_every_experiment: true
model_args:
  hub_id: LiquidAI/LFM2.5-Audio-1.5B
  tts_text: "The quick brown fox jumps over the lazy dog near the river bank."
  tts_max_new_tokens: 96
  asr_seconds: 2.0
protected: [GOAL.md, spec.py, harness/**, .fast-kernel/**, experiments/**, results.tsv]
---

# Goal

Speed up LFM2-Audio (liquid_audio `LFM2AudioModel`, built on the Transformers `Lfm2` backbone, a
FastConformer encoder and a Mimi-compatible 8-codebook decoder). Primary workload: TTS of one sentence
with greedy-equivalent sampling (seeded); secondary: ASR of a 2 s utterance. The returned object must keep
`generate_sequential(**chat, max_new_tokens=...)` semantics.

# Policy (the human decides; the agent never changes it)

- strict: text tokens and audio codes identical to the reference run; decoded waveform SNR >= 40 dB.
- tolerant: >= 90 % identical tokens, SNR >= 20 dB.


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
