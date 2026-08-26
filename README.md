# fast-kernel

**Autoresearch for model inference.** Give it a model — Mimi, LFM2.5, LFM2-Audio, YOLO, or any
torch module — and an agent runs an endless loop of small, measured optimization experiments:
profile → find the slow parts → pick the idea with the largest end-to-end gain → write the kernel
(Triton, TileLang, CuTe DSL, CUDA C++, CUDA graphs, torch.compile, hub kernels) → evaluate against
the frozen reference → keep or revert → learn → repeat. Every experiment builds on the last accepted
one, and every one of them is on a live graph.

It is built for the [Claude Code](https://code.claude.com) CLI: `AGENTS.md` is the research
program, `.claude/agents/` are the roles, `.claude/skills/` are the procedures, and a Stop hook (or
`/loop`, or the headless driver) keeps the session iterating until you say stop. It follows
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) (fixed evaluation, keep/revert,
one editable tree, never stop) and [RightNow-AI/autokernel](https://github.com/RightNow-AI/autokernel)
(profile → Amdahl ranking → five-stage correctness harness → tiered playbook → `results.tsv`), extended
from single kernels to whole models loaded through Transformers.

```
              ┌──────────────── the agent (Claude Code) ────────────────┐
  GOAL.md ──► profile ──► hypothesis ──► edit candidate/ ──► fast-kernel eval ──► note ──┐
              ▲   PLAN.md · ideas        (only tree it may touch)      │                  │
              │                                                        ▼                  │
              │                     harness (immutable): reference oracle → 5 gates →     │
              │                     fixed benchmark → profile → keep (commit, tag exp-N)  │
              │                                       or revert (patch kept)              │
              └──────────────── results.tsv · KNOWLEDGE.md · SQLite events ◄──────────────┘
                                                   │
                                     fast-kernel dashboard (SSE live graph) / report.html
```

## Just give it text

```bash
git clone https://github.com/kadirnar/fast-kernel && cd fast-kernel
uv sync --extra cuda          # torch cu130 + triton + transformers (+ --extra tilelang cute hub yolo audio)
claude
```

then type:

> **/fk-optimize mimi** — or simply: *"Optimize the Mimi codec model and keep improving it."*

The skill scaffolds `campaigns/mimi/`, probes the GPU and every backend, records the baseline
(experiment #0), starts the dashboard, and enters the loop. The same works for `lfm25`, `lfm-audio`,
`yolo`, `custom`, or an existing campaign directory. Stop with `fast-kernel loop stop` (or Esc).

Without an agent, the harness is a normal CLI:

```bash
uv run fast-kernel init mimi                     # campaigns/mimi: GOAL.md, candidate/, PROMPT.md, notes
cd campaigns/mimi
uv run fast-kernel probe                         # GPU roofline + backend compile probes -> capabilities.json
uv run fast-kernel baseline                      # experiment #0 + noise floor + PLAN.md
uv run fast-kernel dashboard --root .. --open    # live graph
#   ... edit candidate/ ...
uv run fast-kernel eval -m "fused RVQ search" --technique fused-quantizer --target t_09d5...
uv run fast-kernel ideas | status | history | show 3 --log | note "..." | report
```

## What happened on the development machine (RTX 5070 Ti, Mimi)

`fast-kernel baseline` on the stock `transformers.MimiModel` (fp32, 1 s of 24 kHz audio, batch 1):
**18.86 ms** round trip, **1641 kernel launches, GPU busy 22.8 % of wall time**. The ranker's top target
was the whole workload (launch bound, Amdahl gain 51 %), then `Conv1d` (37 % of GPU time),
`MimiEuclideanCodebook` (23 % — the 32-stage RVQ search, attributed through its non-`forward`
`quantize()` method), `Linear`, `ConvTranspose1d`, `MimiAttention`.

| # | experiment | status | result |
|--:|---|---|---|
| 0 | baseline (unmodified reference) | baseline | 18.86 ms, 1641 launches |
| 1 | CUDA-graph capture of `encode`/`decode` | crash → reverted | host sync inside `MimiConv1d` (`extra_padding` is a CUDA scalar → `.item()` during capture) |
| 2 | + host-side padding math | crash → reverted | `padding_left/right` are meta tensors after `from_pretrained` |
| 3 | + derive paddings from the int `padding_total` | **keep** | **5.14 ms (3.67×)**, identical codes, waveform allclose, deterministic, GPU busy 97.5 % |
| 4 | *(written autonomously by `fast-kernel auto`)* Triton fp32 implicit-GEMM `Conv1d`, deterministic split-K, for the weight-bound SEANet convs | **keep** | **3.89 ms (4.84×)**, exact codes, 1282 launches |
| 5 | *(the agent's second hypothesis, evaluated after its turn cap)* Triton fp32 fused RVQ codebook search: per-stage exact distance + first-index argmin, fixed-order reductions | **keep** | **3.14 ms (6.0×)**, exact codes, 768 launches |

Experiments 0–3 were run by hand through `fast-kernel eval`; #4 and #5 came from one headless run of
the agent (`fast-kernel auto --iterations 1 --max-turns 40`): it read PLAN.md / KNOWLEDGE.md, chose the top target
(`Conv1d`, 39 % of GPU time after the graphs), wrote a 110-line Triton kernel with exact fp32 FMA and a
fixed-order split-K reduction because the strict policy demands identical codes, verified it per shape,
ran the harness, and recorded the insight ("cuDNN's `precomputed_convolve_sgemm` is 5–10× off the
weight-bandwidth floor on Mimi's small-T convs"). The two crashes cost 6 s each, kept their patches and
tracebacks, and pointed at the exact fix. That is the loop — the next ideas (fused RVQ search,
transformer-block fusion, ConvTranspose) are re-ranked in `PLAN.md` after every keep.

## What is in the box

| piece | where |
|---|---|
| research program for the agent | `AGENTS.md` (imported by `CLAUDE.md`) |
| roles | `.claude/agents/`: fk-orchestrator, fk-profiler, fk-kernel-engineer, fk-verifier, fk-benchmarker, fk-reviewer, fk-librarian |
| procedures | `.claude/skills/`: fk-optimize, fk-experiment, fk-profile, fk-verify, fk-bench, fk-status, fk-dashboard, fk-report, fk-parallel, fk-add-model, fk-loop, hotspot-analysis, triton-kernels, tilelang-kernels, cute-dsl-kernels, cuda-cpp-kernels, cuda-graphs, torch-compile, hub-kernels, numerical-verification |
| loop mechanics | Stop hook `loop_guard.py` (ralph loop with a progress guard), `/loop /fk-experiment`, `fast-kernel auto` (headless `claude -p`), `fast-kernel auto --agents N` (worktree workers + inbox promotion) |
| guard rails | PreToolUse hook `protect_paths.py` (harness files are read-only for the agent), `.claude/settings.json` allow-list |
| harness | `src/fastkernel/harness/`: 5 gates (smoke, shapes, numerical, determinism, edge), fixed benchmark protocol, subprocess isolation, keep/revert with git lineage |
| profiler | `src/fastkernel/profiling/`: module + Python-frame attribution, roofline classification, Amdahl ranking, technique matrix, PLAN.md |
| playbook | `src/fastkernel/playbook.py`: tiers 0-6 (structure, block tuning, memory, fusion, advanced, arch, kernel-specific) |
| backends | `src/fastkernel/backends/`: probes + helpers for Triton, TileLang, CuTe DSL, CUDA C++ (auto nvcc, pip toolchains), torch.compile, CUDA graphs, hub kernels; starter kernels in `templates/` |
| models | `src/fastkernel/models/`: mimi, lfm25 (+ generic HF causal LM), lfm-audio, yolo, custom torch module |
| examples | `examples/{mimi,lfm25,lfm-audio,yolo,custom}/`: GOAL.md, candidate stub, PROMPT.md, recipes |
| live graph | `fast-kernel dashboard` (stdlib HTTP + SSE, zero dependencies), `fast-kernel report` (self-contained HTML) |
| docs | `docs/ARCHITECTURE.md`, `PIPELINE.md`, `DASHBOARD.md`, `MULTI_AGENT.md`, `MODELS.md`, `BACKENDS.md` |

## The experiment contract

- The agent edits **only `candidate/`** (`apply(model, ctx) -> model`, kernels under `candidate/kernels/`).
  `GOAL.md`, `spec.py`, the harness and the ledger are protected (hook-enforced).
- `fast-kernel eval` loads the frozen reference, runs the five gates, the fixed benchmark (warm-up,
  clock ramp, N CUDA-synchronised repeats, median), a profile, and decides: crash → revert; gates
  fail → revert; improvement ≥ max(`min_improvement`, measured noise floor) → **keep** (commit on
  `fast-kernel/<model>`, tag `exp-N`, incumbent moves, PLAN.md re-ranked); else revert (`--simpler`
  keeps equal-speed simplifications, as in autoresearch).
- Everything is recorded: `results.tsv`, `experiments/NNNN-*/` (metrics, gates, profile, log, patch),
  `KNOWLEDGE.md`, SQLite events for the dashboard.
- **No hardware limitations**: `capabilities.json` is evidence. When a backend does not compile, the
  agent fixes the environment (`fast-kernel toolchain install --cuda 13.3` installs a self-contained
  nvcc/CCCL/NVVM set from pip wheels — this is how TileLang and CUDA C++ became READY on a gcc 16
  machine) or switches backend; it never concludes "unsupported".
- **Never stop**: plateaus change the technique tier, the backend, the target, or the scope; the loop
  ends only when a human says so.

## Models

| name | loaded through | primary workload | strict gate |
|---|---|---|---|
| `mimi` | `transformers.MimiModel` | encode+decode 1 s @ 24 kHz | identical codes, waveform allclose |
| `lfm25` | `AutoModelForCausalLM` (`Lfm2ForCausalLM`), variants 1.2B/350M/230M | greedy decode 64 tokens, prefill 512 | identical tokens, top-1 ≥ 99.5 % |
| `lfm-audio` | `liquid_audio` on the Transformers `Lfm2` backbone + Mimi decoder | TTS one sentence, ASR 2 s | identical tokens, SNR ≥ 40 dB |
| `yolo` | `ultralytics.YOLO(...).model` (fused torch module) | detect batch 1 / 8 @ 640² | boxes ≤ 0.5 px, conf ≤ 1e-3 |
| `custom` | `spec.py: build_model()` | forward | allclose |

`fast-kernel init <name> --set model_args.variant=350m --precision tolerant` tunes a campaign;
`/fk-add-model` onboards anything else.

## Multi-agent

`fast-kernel auto --agents 3` runs three workers on git worktrees, each leasing a hotspot; their
accepted patches go to an inbox and are re-measured on top of the incumbent one by one, so the lineage
stays a single chain of verified improvements. In-session, the orchestrator delegates to the
profiler / kernel-engineer / verifier / reviewer / librarian subagents. See `docs/MULTI_AGENT.md`.

## Development

```bash
uv sync --extra cuda --extra dev
uv run pytest -q            # CPU-only tests (config, ranking, store, campaign/git, hooks, report)
uv run fast-kernel doctor
```

The core is stdlib-only (torch is imported lazily inside the harness), so `status`, `dashboard`,
`report` and the hooks work anywhere. MIT license.
