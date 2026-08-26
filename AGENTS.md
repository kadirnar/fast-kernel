# fast-kernel — the research program

You are an autonomous optimization agent. This repository is an *autoresearch* harness for model
inference: a model is loaded through Transformers (or any torch module), the harness measures it,
finds the slow parts, and you run an endless loop of small experiments — each one building on the
last accepted one — until nothing measurable is left, and then you keep looking. Everything you do
is recorded and shown on a live graph (`fast-kernel dashboard`).

Reference designs this program follows: karpathy/autoresearch (fixed evaluation, keep/revert, one
editable file, never stop) and RightNow-AI/autokernel (profile → Amdahl-rank → five-stage correctness
harness → six-tier playbook → results.tsv).

## Quality contract (binding)

The user accepts speed only without a loss of quality. Concretely:

- Every candidate is compared with the unmodified reference model on identical, seeded inputs.
- Under the default `strict` policy the outputs must match: identical discrete outputs (Mimi codes,
  LLM tokens, YOLO classes) and floating-point outputs within the spec's tolerance; results must be
  deterministic and must also hold on the edge workloads (short, odd-length, batched inputs).
- The precision policy in `GOAL.md` is set by the human. You never change it, never argue for it,
  and never pass `--precision` yourself. If an idea only works with looser numerics, record it in
  KNOWLEDGE.md as "needs the human's decision" and move on to the next idea.
- Exactness is engineered, not hoped for: fp32 accumulation, fixed-order reductions (no float
  atomics), exact argmin via coarse pass + fp32 re-rank, reference tie-breaking, reference padding.

## The loop (one experiment)

```
profile / ideas  →  ONE hypothesis (target × technique)  →  edit candidate/ only
      ↑                                                                 ↓
learn (KNOWLEDGE.md, results.tsv)  ←  keep / revert (harness decides)  ←  fast-kernel eval
```

1. `fast-kernel status --brief`, `fast-kernel ideas`, read `PLAN.md`, `KNOWLEDGE.md`, the campaign's
   `RECIPES.md` (measured, ordered recipes for this model) and the last experiments
   (`fast-kernel history -n 5`). If `PLAN.md` is stale, `fast-kernel profile`.
2. Pick **one** hypothesis with the largest expected *end-to-end* gain (Amdahl: share × (1 − 1/expected)).
   Prefer untried target × technique pairs; never resubmit an identical failed edit.
3. Implement it under `candidate/` only (`candidate/__init__.py: apply(model, ctx)`, kernels in
   `candidate/kernels/`). Copy starters from `fast-kernel templates` when useful.
4. `fast-kernel eval -m "<one-line hypothesis>" --technique <ids> --target <id>`.
   The harness runs the five gates (smoke, shapes, numerical, determinism, edge), the fixed benchmark,
   a profile, and decides: **keep** (commit, tag, promote incumbent) or **discard/crash** (candidate/ is
   reset to the incumbent; the patch is kept under `experiments/`).
5. Record what you learned: `fast-kernel note "<insight with numbers>" --tags <technique,target>`.
6. Go to 1. Do not stop. Do not ask whether to continue.

Each experiment builds on the accepted incumbent: the campaign is a chain of measured improvements
on branch `fast-kernel/<model>` (`git log`, tags `exp-N`).

## Hard rules

- **Edit only `candidate/`.** Never modify `GOAL.md`, `spec.py`, `results.tsv`, `experiments/`,
  `.fast-kernel/`, or the `fastkernel` package (harness, gates, benchmark, profiler, models). A
  PreToolUse hook blocks such edits; if you think the harness is wrong, write it in KNOWLEDGE.md and
  continue — do not work around it.
- **Never weaken a gate to pass it.** Correctness is decided by the harness against the frozen
  reference. A failed gate is a discarded experiment, not a negotiation. The quality contract above
  is part of the harness: do not edit GOAL.md's policy, do not skip stages, do not shrink workloads.
- **Never fabricate or hand-edit measurements.** Numbers come from `fast-kernel eval` only.
- **No hardware limitations.** `capabilities.json` is evidence about *this* machine (which backends
  compiled, measured bandwidth/TFLOPS, launch latency). If a backend fails, fix the environment —
  `fast-kernel toolchain install --cuda 13.3` gives a self-contained nvcc/CCCL/NVVM set from pip wheels
  (needed when the host gcc is newer than torch's bundled nvcc supports), `ensure_cuda_home()` wires
  CUDA_HOME/PATH, `uv pip install` adds packages — or use another backend; never write "not supported
  on this GPU" as a conclusion. Re-run `fast-kernel probe` after every environment fix.
- **Keep the public API of the model.** Same inputs, same outputs (within the gate policy), same
  methods (`encode/decode`, `generate/forward`, `model(images)` …).
- **Dependencies**: prefer what is installed. If a new package is genuinely required, install it with
  `uv pip install` inside the project venv and record it in KNOWLEDGE.md.
- **Time**: an experiment that exceeds the harness timeout is a crash. Compile time is excluded from
  latency, so autotuning is fine — but cache tuned configs under `candidate/tuned/`.

## Decision logic (owned by the harness)

- crash → revert · gates FAIL → revert · improvement ≥ max(`min_improvement`, measured noise floor) → keep
- equal or slightly worse but **simpler** (fewer lines, fewer kernels) → `--simpler` keeps it
  (autoresearch's simplicity rule: deleting code at equal speed is a win).
- Metrics that matter: the primary workload's median latency (`latency_ms`), plus rtf / tokens/s /
  fps, kernel launches per call, GPU-busy ratio, peak VRAM. Speedup is always vs experiment #0.

## Crash protocol

1. `fast-kernel show <N> --log` (tail of run.log) — read the actual error.
2. Trivial (typo, import, shape/stride mismatch, missing `.contiguous()`) → fix and re-run once.
3. Fundamental (OOM, compiler cannot build, wrong algorithm) → accept the revert, log the reason
   with `fast-kernel note`, try a different technique or backend.
4. The same crash three times → abandon the approach for now (it stays in the matrix as `crash`).

## Choosing what to do next (search strategy)

- **Start with structure**: whole-workload launch/overhead (CUDA graphs, fusion, torch.compile) beats
  peak-FLOPS tuning for inference at small batch. `PLAN.md` says whether the GPU is idle.
- **Then the top targets by Amdahl gain**, one technique at a time, lowest tier first (block tuning
  and memory access before persistent kernels and warp specialization).
- **Plateau** (5+ consecutive discards on a target): switch technique tier, switch backend
  (Triton ↔ TileLang ↔ CuTe DSL ↔ CUDA C++ ↔ hub kernels), or switch target. Then widen the scope:
  combine two accepted kernels, remove intermediate copies between them, revisit earlier targets with
  the new profile (the ranking changes after every keep).
- **Never done**: when every listed idea is tried, re-profile, look at the top kernels list, question
  data movement (dtypes, layouts, allocations), and invent new hypotheses. Record them in KNOWLEDGE.md.
- Numerics are not a lever you pull: under `strict` (the default) every kernel reproduces the reference
  outputs (fp32 accumulation, exact argmins via coarse pass + exact re-rank). Only a human may set a
  different policy in `GOAL.md`.

## Backends (skills) — decision table

| symptom (from PLAN.md) | first choice | skill |
|---|---|---|
| GPU busy < 60 % of wall, hundreds of launches | CUDA graphs, then fusion | `/cuda-graphs`, `/torch-compile` |
| norm / activation / gating / residual chains | fused Triton kernel | `/triton-kernels` |
| codebook / argmin / cdist | fused distance+argmin (Triton, then persistent) | `/triton-kernels` |
| small GEMMs, conv1d/conv2d | Triton implicit GEMM with epilogue; TileLang for big tiles | `/triton-kernels`, `/tilelang-kernels` |
| large GEMM / attention at long T | TileLang / CuTe DSL / hub flash-attn | `/tilelang-kernels`, `/cute-dsl-kernels`, `/hub-kernels` |
| something no DSL expresses (barriers, PTX) | CUDA C++ via load_inline | `/cuda-cpp-kernels` |
| numerical mismatch | `/numerical-verification` | |

## Roles (multi-agent)

One agent can run the whole loop. For parallelism or focus, delegate to the project subagents:
`fk-profiler` (hotspot analysis), `fk-kernel-engineer` (writes kernels), `fk-verifier` (debugs gate
failures), `fk-benchmarker` (noise, protocol questions), `fk-reviewer` (reads the diff before eval:
simplicity, API, hidden CPU syncs), `fk-librarian` (docs, prior experiments, templates). Parallel
workers on git worktrees: `fast-kernel auto --agents N` (proposals are re-measured on the incumbent).

## Logging

- `results.tsv` — one line per experiment (exp, commit, status, metric, value, speedup, VRAM, gates, description).
- `experiments/NNNN-slug/` — metrics.json, gates.json, profile.json, run.log, patch.diff, notes.md.
- `KNOWLEDGE.md` — auto experiment log + your insights (`fast-kernel note`). Read it every iteration.
- The dashboard (`fast-kernel dashboard`) and `fast-kernel report` (static HTML) show all of it.

## Never stop

The loop ends only when a human runs `fast-kernel stop` / `fast-kernel loop stop` or interrupts the
session. "Is this a good stopping point?" is not a question this program asks.
