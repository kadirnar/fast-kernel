# fast-kernel

fast-kernel makes a model run faster **without changing what it produces**. You give an agent
(Claude Code) the prompt below; it profiles the model, finds the slow parts, writes and tests GPU
kernels (Triton, TileLang, CuTe DSL, CUDA C++, CUDA graphs, torch.compile), measures each attempt
against the original model, keeps only what is faster *and* identical in output, and repeats for as
long as you let it — with several agents working at once and every experiment on a live graph.

## Setup (once)

```bash
git clone https://github.com/kadirnar/fast-kernel && cd fast-kernel
uv sync --extra cuda          # add --extra audio for LFM2-Audio, --extra yolo for YOLO
claude                        # start Claude Code inside this folder
```

You need an NVIDIA GPU, Python 3.12+, [uv](https://docs.astral.sh/uv/) and [Claude Code](https://code.claude.com).

## The prompt (Mimi codec)

Paste this into Claude Code. The first line alone already works — `CLAUDE.md` routes it to the
`fk-optimize` skill, which reads the same files and runs the same steps — but the full text tells the
agent exactly what to read, how the library works, how to use the other agents, and what quality means.

```text
Optimize the Mimi codec model.

Optimize the Mimi neural audio codec (`kyutai/mimi`, loaded through `transformers.MimiModel`) with
fast-kernel: make encode + decode of one second of 24 kHz audio (batch 1, the primary workload
`roundtrip_1s`) as fast as possible on this GPU, and keep improving it for as long as I let you run.
Speed is only accepted without any loss of quality: the optimized model must produce exactly the same
audio codes and the same waveform as the original.

## What to read before doing anything (in this order)

1. `README.md`, `AGENTS.md` (the research program — every rule below is binding) and `CLAUDE.md`.
2. `.claude/agents/*.md` (the roles you may delegate to) and these skills in `.claude/skills/`:
   `fk-optimize`, `fk-experiment`, `fk-parallel`, `hotspot-analysis`, `numerical-verification`, and the
   backend skills `cuda-graphs`, `triton-kernels`, `tilelang-kernels`, `cute-dsl-kernels`,
   `cuda-cpp-kernels`, `torch-compile`, `hub-kernels`.
3. The harness you must never edit but must understand: `src/fastkernel/harness/gates.py` (the five
   correctness stages), `src/fastkernel/harness/evaluate.py` (keep/revert logic), `src/fastkernel/harness/run.py`,
   `src/fastkernel/profiling/` (how targets are ranked), `src/fastkernel/playbook.py` (technique ids),
   `src/fastkernel/backends/templates/` (starter kernels) and `src/fastkernel/backends/graphs.py`.
4. The model spec: `src/fastkernel/models/mimi.py` — it defines the reference oracle, the workloads, the exact
   correctness checks and the hotspot hints. Read the model's own source too (`transformers/models/mimi/modeling_mimi.py` in site-packages).
5. The campaign, once it exists: `campaigns/mimi/GOAL.md` (objective, metric, quality policy),
   `PLAN.md` (ranked targets with shapes and technique status), `RECIPES.md` (measured, ordered recipes),
   `KNOWLEDGE.md` (insights and the experiment log), `results.tsv`, and `experiments/NNNN-*/` (metrics,
   gates, profile, patch, log of every experiment).

## How the library works (do it in this order; every step is idempotent)

1. `uv run fast-kernel resolve "<this sentence>" --json` — maps the sentence to the model, to the
   absolute campaign folder `campaigns/mimi` (created the first time, reused later) and to the
   remaining steps. Run every later command as `cd <campaign> && uv run fast-kernel ...`.
2. `uv run fast-kernel doctor` — torch/CUDA/backends/claude present; apply its fix hints
   (`uv sync --extra cuda`).
3. `uv run fast-kernel init mimi` (only if the campaign does not exist).
4. `uv run fast-kernel probe` — GPU roofline numbers and one compiled probe kernel per backend
   (Triton, TileLang, CuTe DSL, CUDA C++, torch.compile, CUDA graphs, hub kernels) → `capabilities.json`.
   A backend that fails with a host-compiler/nvcc error is fixed with
   `uv run fast-kernel toolchain install --cuda 13.3` and probed again — never recorded as a hardware limit.
5. `uv run fast-kernel baseline` — experiment #0: the unmodified reference model passes its own five
   gates, is benchmarked with the fixed protocol (warm-up, clock ramp, median of N CUDA-synchronised
   runs), the noise floor is measured, and the profile ranks the targets into `PLAN.md`.
6. `uv run fast-kernel dashboard --root campaigns` in the background — the live graph of every
   experiment (it prints the URL; tell me once).
7. `uv run fast-kernel loop start` — the Stop hook keeps this session iterating.
8. The loop, forever: `fast-kernel status --brief` → `fast-kernel ideas` → read PLAN.md / KNOWLEDGE.md /
   RECIPES.md → one hypothesis with the largest end-to-end gain (share x (1 - 1/expected), untried
   first, recipe order, lower tier first) → implement it only under `candidate/`
   (`candidate/__init__.py: apply(model, ctx)`, kernels in `candidate/kernels/`) → self-test on the real
   shapes → `fast-kernel eval -m "<one line>" --technique <ids> --target <id>` → read the verdict →
   `fast-kernel note "<insight with numbers>"` → next. Each experiment builds on the accepted incumbent
   (git branch `fast-kernel/mimi`, tags `exp-N`); rejected ones are reverted, their patches kept.

## Multi-agent structure (use it)

You are the orchestrator (`fk-orchestrator`). Delegate with the Agent tool:
- `fk-profiler` after the baseline and after every surprising result: explain where the time goes
  (launch-bound vs compute vs memory), re-rank, name the next hypothesis.
- `fk-kernel-engineer`, one per target, in parallel: `uv run fast-kernel worktree create eng-<target>`
  gives each a private worktree branched from the incumbent; each engineer writes its kernel there,
  self-tests, and ends with `fast-kernel propose -m "..." --technique <id> --target <id>`. Then you run
  `uv run fast-kernel inbox`, which applies and measures every proposal serially on the incumbent
  (explore in parallel, measure serially — one GPU, one lineage). Details: `/fk-parallel`.
- `fk-verifier` on every failed gate (root cause and the minimal exact fix), `fk-reviewer` before an
  expensive evaluation (API preserved? hidden host syncs? simpler version?), `fk-benchmarker` when
  numbers look noisy, `fk-librarian` for references, templates and prior experiments.
- Alternatively run headless workers: `uv run fast-kernel auto --agents 3` (worktrees, hotspot leases,
  inbox promotion) and keep watching the dashboard.

## Rules (from AGENTS.md; the harness enforces them, you follow them)

- Edit only `candidate/`. Never touch `GOAL.md`, `spec.py`, `results.tsv`, `experiments/`,
  `.fast-kernel/` or the `fastkernel` package (a hook blocks it). Never weaken, skip or shrink a gate,
  a workload or the policy; never hand-edit measurements.
- Quality contract: outputs must match the original model — identical discrete audio codes, decoded waveform allclose (rtol 2e-4, atol 2e-5), identical results on 0.25 s / 5 s / noise inputs and on the 50 ms, odd-length and batch-2 edge cases. Under the default `strict`
  policy this is checked on every workload, deterministically, and on the edge inputs. You never change
  the policy; if an idea needs looser numerics, write "needs the human's decision" in KNOWLEDGE.md and move on.
- No hardware limitations: `capabilities.json` is evidence. Missing packages, headers, nvcc, weights →
  install/fix (`uv pip install`, `fast-kernel toolchain install`), re-probe, continue.
- Crashes: `fast-kernel show <N> --log`; trivial (typo, import, stride, meta tensor) → fix and rerun
  once; fundamental → note it and take the next idea; the same crash three times → abandon the approach.
- Plateaus (5 discards on a target): change technique tier, then backend
  (Triton ↔ TileLang ↔ CuTe DSL ↔ CUDA C++ ↔ hub kernels), then target; then widen the scope (combine
  kept kernels, remove copies between them); re-profile. The list of ideas is never empty.
- Report as short lines: `#N kept/discarded/crashed · <metric> (Δ) · <speedup>x vs baseline · next: <idea>`.
  Never end a turn with a question. Never ask whether to continue.

## Stopping

Only I stop this: when I write "Stop optimizing." run `fast-kernel loop stop` (and `fast-kernel stop`
if headless workers run). The campaign is persisted; the same sentence later continues it.

## Where the time goes and what to try (measured on an RTX 5070 Ti; re-measure here)

The stock model is launch-bound (1641 kernels for 1 s of audio, GPU busy 23 %, 18.86 ms). The measured
path so far: CUDA graphs for encode/decode with host-side conv padding math (5.14 ms), a Triton fp32
implicit-GEMM Conv1d with deterministic split-K (3.89 ms), a Triton fp32 fused RVQ codebook search
(3.14 ms, 6x) — all with identical codes. Next, in order of expected payoff (see RECIPES.md):
ConvTranspose1d and the remaining stride-1 convs as implicit GEMMs with fused ELU epilogues;
transformer-block fusion (LN+QKV+RoPE, attention+O+residual+LayerScale, LN+FC1+GELU, FC2) for the
16 latency-bound layers; launch-count reduction inside the graphs (copies, casts, pads, ELU);
a tensor-core coarse pass + exact fp32 re-rank for the RVQ distances (codes stay identical); a persistent
32-stage RVQ kernel only if measured faster than per-stage launches. bf16 in the quantizer is a human
decision (it flips near-tie codes) — do not take it.
```

To stop: type `Stop optimizing.` — the campaign is saved; the same prompt later continues it.

## Other models

The same prompt with a different first paragraph, ready to paste:
[`examples/lfm25/PROMPT.md`](examples/lfm25/PROMPT.md) (LFM2.5),
[`examples/lfm-audio/PROMPT.md`](examples/lfm-audio/PROMPT.md) (LFM2-Audio),
[`examples/yolo/PROMPT.md`](examples/yolo/PROMPT.md) (YOLO),
[`examples/custom/PROMPT.md`](examples/custom/PROMPT.md) (your own PyTorch model).

## What you will see

- `campaigns/mimi/results.tsv` — one line per experiment (kept / discarded / crashed, with numbers)
- `campaigns/mimi/experiments/NNNN-*/` — metrics, correctness checks, profile, patch and log of each
- `campaigns/mimi/KNOWLEDGE.md` — what the agents learned; `PLAN.md` — the current ranked targets
- the dashboard (URL printed at start; `uv run fast-kernel report` writes the same graph to one HTML file)

Measured on an RTX 5070 Ti: Mimi encode+decode of one second of audio went from 18.9 ms to 3.1 ms
(6x) in five experiments, with identical codes throughout; two of those kernels were written by the
agent unattended.

## Quality

Every experiment is compared with the unmodified original model on the same inputs: Mimi must give the
same codes and waveform, language models the same tokens, YOLO the same boxes and classes; results
must be deterministic and hold on short, odd-length and batched inputs. An experiment that changes
outputs is discarded automatically, however fast it is. The checks live outside the code the agents
may edit, and the agents may not change the policy.

## How it works

`docs/ARCHITECTURE.md`, `docs/PIPELINE.md`, `docs/MULTI_AGENT.md`, `docs/DASHBOARD.md`,
`docs/MODELS.md`, `docs/BACKENDS.md`. MIT license.
