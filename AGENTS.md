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

1. `fast-kernel status --brief`, `fast-kernel ideas`, read `PLAN.md`, `KNOWLEDGE.md` and the last experiments
   (`fast-kernel history -n 5`). If `PLAN.md` is stale, `fast-kernel profile`.
2. Pick **one** target — the one with the largest measured share of end-to-end time — and form **one**
   hypothesis for it. Which technique/backend to try is yours to discover from the measurements (nothing
   tells you the method or how much it will help). Prefer untried target × approach pairs; never resubmit
   an identical failed edit.
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
  (needed when the host compiler is newer than torch's bundled nvcc supports), `ensure_cuda_home()` wires
  CUDA_HOME/PATH, `uv pip install` adds packages — or use another backend; never write "not supported
  on this GPU" as a conclusion. Re-run `fast-kernel probe` after every environment fix.
- **Keep the public API of the model.** Same inputs, same outputs (within the gate policy), same
  methods (`encode/decode`, `generate/forward`, `model(images)` …).
- **Dependencies**: prefer what is installed, but a missing library is never a blocker and never a
  reason to ask a human. Install it yourself with `uv pip install` inside the project venv (the harness
  also auto-installs a backend's package on probe) and record it in KNOWLEDGE.md.
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

- **Measure, then decide.** `PLAN.md` and `fast-kernel ideas` give you *measured facts only* — each
  target's share of GPU time, its **roofline efficiency (SOL%)** = how close it already runs to this
  machine's measured bandwidth / FLOP-per-second peak, its boundness, kernel counts, and what earlier
  experiments already tried. They deliberately do **not** name a technique or predict a speedup: which
  backend and which transformation to try is yours to discover from the profile, the top-kernels list,
  KNOWLEDGE.md and the backend skills. Never assume a method wins before you have measured it.
- **Pick the target with the most measured headroom** = share × (1 − SOL). A target that is a big share
  of time but already near its roofline (high SOL) has little left to give; a low-SOL target is where
  real speed hides. The whole-workload roofline efficiency in PLAN.md tells you how close the model as a
  whole is to the hardware limit. Change one thing, measure, keep or revert. Prefer untried target ×
  approach pairs; never resubmit an identical failed edit.
- **Plateau** (5+ consecutive discards on a target): switch approach, switch backend, or switch target,
  then widen the scope — combine two accepted kernels, remove intermediate copies, and revisit earlier
  targets (the ranking changes after every keep).
- **Never done**: when the obvious ideas are tried, re-profile, look at the top kernels, question data
  movement (dtypes, layouts, allocations), and invent new hypotheses. Record them in KNOWLEDGE.md.
- Numerics are not a lever you pull: under `strict` (the default) every kernel reproduces the reference
  outputs (fp32 accumulation, exact argmins via coarse pass + exact re-rank). Only a human may set a
  different policy in `GOAL.md`. Never distillation, retraining or fine-tuning — the outputs stay the
  frozen reference model's.

## Backends (skills) — references, not prescriptions

These skills document how to implement kernels on each backend. They are references you reach for once
*your* measurement points you at a target — nothing here maps a symptom to a mandated method:

- `/cuda-graphs`, `/torch-compile` — capturing / fusing a launch-bound workload
- `/triton-kernels` — fused elementwise/norm/gating chains, epilogue-fused GEMMs, implicit-GEMM convs,
  codebook argmin, fused attention, persistent kernels
- `/tilelang-kernels`, `/cute-dsl-kernels`, `/hub-kernels` — pipelined GEMM/attention, hand-scheduled
  tiles, pre-built hub kernels
- `/cuda-cpp-kernels` — anything a DSL cannot express (barriers, PTX, warp specialization)
- `/numerical-verification` — keeping a kernel bit-faithful to the reference

A backend that is not installed is not a limit: install it (`uv pip install ...`, the harness also does
this automatically on probe) or use `fast-kernel toolchain install`, then continue.

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

The loop runs autonomously until the optimization is **exhausted** — a measured convergence signal:
`CONVERGE_AFTER` consecutive experiments that ran but improved nothing. At that point the loop stops
itself (the `loop.active` flag is cleared). Until then it keeps profiling, hypothesising, evaluating and
learning. It never asks the human whether to continue, and it never asks the human to run a stop command
— "Is this a good stopping point?" is not a question this program asks. A human may still stop it early
with `fast-kernel stop` / `fast-kernel loop stop`, but the program never solicits that.
