# Model specs

The default policy everywhere is `strict`: outputs identical to the original model. `tolerant` exists
for a human who explicitly accepts numerically different but quality-preserving outputs; the agent never
selects it.

A spec (`fastkernel.models.ModelSpec`) answers five questions; the harness does the rest.

| method | answer |
|---|---|
| `load_reference()` | the frozen oracle: eval mode, on CUDA, deterministic (seeds, TF32 off) |
| `load_candidate(ctx)` | default: `load_reference()` then `candidate.apply(model, ctx)` |
| `workloads()` | `Workload(name, make_inputs(device, seed), run(model, inputs), primary, bench, tags, units)` |
| `compare(workload, ref, cand)` | model-specific `GateCheck`s (default: allclose at the policy tolerance) |
| `hotspot_hints()` / `notes` | class -> category hints and architecture notes for PLAN.md |

`units` (`audio_seconds`, `tokens`, `images`, `samples`) turn latency into rtf / tokens per s / fps.
`tags=("sweep",)` workloads are checked and benchmarked; `tags=("edge",)` workloads only checked.

## Built-ins

- **mimi** (`transformers.MimiModel`, `kyutai/mimi`): roundtrip/encode/decode at 1 s (+0.25 s, 5 s
  sweeps, noise input), edge: 50 ms, odd length, batch 2. Strict (default): identical codes + waveform allclose
  (rtol 2e-4 / atol 2e-5); tolerant (human opt-in): decode SNR >= 40 dB, reconstruction SNR within
  0.25 dB of the reference on every workload, >= 80 % identical codes. Hints cover the codebook search, attention, MLP, convs, LayerScale.
- **lfm25** / **hf-causal-lm** (`Lfm2ForCausalLM` via `AutoModelForCausalLM`): decode (64 greedy tokens
  after a 64-token chat-templated prompt; primary) and prefill (512 tokens, last-8 logits). Strict (default):
  identical tokens, top-1 >= 99.5 %, top-5 overlap >= 0.9; tolerant (human opt-in): >= 98 % identical
  tokens, top-1 >= 99 %. `variant` selects
  1.2B-Instruct/Base/Thinking, 350M, 230M.
- **lfm-audio** (`liquid_audio.LFM2AudioModel`, LFM2.5-Audio-1.5B / LFM2-Audio-1.5B): TTS of one
  sentence (greedy-equivalent sampling, seeded) -> codes + Mimi-decoded waveform (primary), ASR of a
  synthetic 2 s utterance (or `asr_audio` wav). Strict (default): identical tokens, SNR >= 40 dB; tolerant (human opt-in): >= 98 % tokens, SNR >= 30 dB.
- **yolo** (Ultralytics `YOLO(weights).model`, fused): detect batch 1 @ 640 (primary) and batch 8;
  edge 320 px and batch 3. End-to-end heads compare boxes (<= 0.5 px strict), confidences (1e-3) and
  classes for detections with conf > 0.25; raw heads use allclose.
- **custom** (`spec.py: build_model()`, `TorchModuleSpec`): random inputs from `input_shapes`,
  optional `batch_sweep`, allclose gates.

## Adding a model

`fast-kernel init custom --name <x>`, edit `spec.py` (or subclass a built-in), set `model:` and
`model_args:` in GOAL.md, run `fast-kernel baseline`. The `/fk-add-model` skill walks an agent
through it. To make it a first-class template, add `fastkernel/templates/models/<x>/` and an entry
in `fastkernel/models/registry.py`.
