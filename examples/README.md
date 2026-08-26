# Examples

Each directory is the exact campaign template that `fast-kernel init <name>` copies into
`campaigns/<name>/`: a `GOAL.md` (objective, metric, gate policy, benchmark protocol, model args),
the agent-owned `candidate/` package (identity `apply(model, ctx)` to start from), a `PROMPT.md` with
the one paragraph you give Claude Code, and model notes.

| example | model | loaded through | primary workload | strict gate |
|---|---|---|---|---|
| `mimi/` | `kyutai/mimi` neural audio codec | `transformers.MimiModel` | encode+decode 1 s @ 24 kHz, batch 1 | identical codes, waveform allclose |
| `lfm25/` | `LiquidAI/LFM2.5-1.2B-Instruct` (also 350M/230M/Base/Thinking) | `transformers.AutoModelForCausalLM` (`Lfm2ForCausalLM`) | greedy decode 64 tokens (+ prefill 512) | identical tokens, top-1 ≥ 99.5 % |
| `lfm-audio/` | `LiquidAI/LFM2.5-Audio-1.5B` (LFM2-Audio) | `liquid_audio` on the Transformers `Lfm2` backbone + Mimi decoder | TTS one sentence (+ ASR 2 s) | identical text/audio tokens, SNR ≥ 40 dB |
| `yolo/` | Ultralytics YOLO26n (any `.pt`) | `ultralytics.YOLO(...).model` fused torch module | detect batch 1 @ 640² (+ batch 8) | boxes ≤ 0.5 px, conf ≤ 1e-3, same classes |
| `custom/` | any torch module | `spec.py: build_model()` | forward | allclose |

The model, the reference oracle, the workloads, the gates and the profiler hints live in
`src/fastkernel/models/<name>.py`; the campaign only carries the goal and the candidate.

## Just give it text

```bash
uv sync --extra cuda            # + --extra yolo / --extra audio / --extra tilelang / --extra cute / --extra hub
claude                          # in the repository root
```

then type, for example:

> `/fk-optimize mimi` — or simply: *"Optimize the Mimi codec model and keep improving it."*

The skill scaffolds the campaign, probes the GPU and backends, records the baseline, starts the live
dashboard and enters the endless experiment loop. `fast-kernel loop stop` ends it.
