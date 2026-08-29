# fast-kernel — the research program

You are an autonomous optimization agent. This repository is an *autoresearch* harness for model
inference: a model is loaded through Transformers (or any torch module), the harness measures it,
finds the slow parts, and you run an endless loop of small experiments — each one building on the
last accepted one — until nothing measurable is left, and then you keep looking. Everything you do
is recorded and shown on a live graph (`fast-kernel dashboard`).

Reference designs this program follows: karpathy/autoresearch (fixed evaluation, keep/revert, one
editable file, never stop), RightNow-AI/autokernel (profile → rank → five-stage correctness harness →
results.tsv) and the hardware-guided agent loops of KernelAgent / KernelSkill / AKG (roofline
diagnosis, reflexion + repair memory, population search). Targets are ranked by *measured* headroom —
share of GPU time × (1 − roofline efficiency) — never by a predicted speedup.

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

1. `fast-kernel brief` — one screen: incumbent and threshold, the **plateau streak** (consecutive experiments
   without a measured improvement, and how many of them on the same target), the ranked targets with their
   measured memory (what worked, which failures not to repeat), the last experiments and the latest insights.
   Go deeper only where needed: `PLAN.md` (shapes, top kernels), `KNOWLEDGE.md` (every insight),
   `fast-kernel memory --target <id>`, `fast-kernel show <N> --log`. If `PLAN.md` is stale, `fast-kernel profile`.
2. Pick **one** target — the one with the largest measured share of end-to-end time — and form **one**
   hypothesis for it. Which technique/backend to try is yours to discover from the measurements (nothing
   tells you the method or how much it will help). Prefer untried target × approach pairs; never resubmit
   an identical failed edit.
3. Implement it under `candidate/` only (`candidate/__init__.py: apply(model, ctx)`, kernels in
   `candidate/kernels/`). Copy starters from `fast-kernel templates` when useful.
4. `fast-kernel eval -m "<one-line hypothesis>" --technique <ids> --target <id>`.
   The harness runs the five gates (smoke, shapes, numerical, determinism, edge), the fixed benchmark
   plus an interleaved comparison against the reference model, a profile, and decides: **keep**
   (commit, tag, promote incumbent), **bank** (a real gain too small to resolve on its own: committed
   and left in `candidate/` for the next experiment to build on) or **discard/crash** (candidate/ is
   reset; the patch is kept under `experiments/`).
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

- crash → revert · gates FAIL → revert · improvement ≥ max(`min_improvement`, measurement
  uncertainty) → **keep** (commit, tag, promote incumbent).
- improvement > 0 but below that threshold → **bank**: the harness commits your work and leaves it
  in `candidate/`, but does not move the incumbent. The next experiment builds on top of it, and
  the accumulated tree is promoted as one keep as soon as the pile is jointly large enough to
  measure. A bank is a success, not a rejection — a 0.4 % win that used to be discarded is now
  kept. Up to `bench.max_banked` (default 8) may pile up before the harness stops banking.
- equal or slightly worse but **simpler** (fewer lines, fewer kernels) → `--simpler` keeps it
  (autoresearch's simplicity rule: deleting code at equal speed is a win).
- **How "improvement" is measured.** Absolute milliseconds are not comparable between sessions:
  clocks, thermals and whatever else touches the GPU drift, and a raw comparison charges that drift
  to your candidate. So every experiment times the candidate **and the unmodified reference model
  interleaved, in one process** (`metrics.anchor`), and the decision uses the ratio of ratios. The
  threshold is that measurement's own uncertainty, recomputed per experiment — not a number frozen
  at baseline. A verdict that lands within its own uncertainty of a boundary (keep/bank or
  bank/discard) is measured longer automatically — more interleaved pairs, up to `bench.anchor_max_pairs`
  — until it is resolved; a clear win or loss costs one batch. `fast-kernel show <N>` prints which basis
  was used and how many pairs it took.
- Because of this, **do not chase the threshold by bundling unrelated edits**. One hypothesis per
  experiment stays correct: small wins accumulate through banking, not by hiding several ideas in
  one diff where you cannot tell which one paid.
- Metrics that matter: the primary workload's median latency (`latency_ms`), plus rtf / tokens/s /
  fps, kernel launches per call, GPU-busy ratio, peak VRAM. Speedup is always vs experiment #0.
- A campaign that predates anchoring has no reference ratio for its incumbent. One
  `fast-kernel eval --force` with an unchanged tree re-measures the incumbent, records its anchor
  and counts as neither progress nor failure; every experiment after that gets the precise
  comparison.

## Crash protocol

Every finished experiment is auto-classified into a **failure class** (compile, import, shape,
illegal-memory, oom, timeout, numerical, determinism, edge, nan, …) shown by `fast-kernel show <N>`
and stored in the campaign memory. Route by that class (adaptive error routing): an `import` class
means install the package and retry; `shape` / `illegal-memory` means fix strides/bounds/`.contiguous()`;
`numerical` / `determinism` means debug against the reference (`/numerical-verification`); `oom` /
`timeout` means a smaller working set or a different algorithm. The class is a *diagnosis*, not a fix.

1. `fast-kernel show <N> --log` (tail of run.log) — read the actual error and the failure class.
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
- **Plateau** — `fast-kernel brief` / `status` print the streak: consecutive experiments without a measured
  improvement, and how many of them on the same target. At 5+ it says PLATEAU: switch approach, switch
  backend, or switch target, then widen the scope — combine two accepted kernels, remove intermediate copies,
  and revisit earlier targets (the ranking changes after every keep). The streak is measured, not felt.
- **Never done**: when the obvious ideas are tried, re-profile, look at the top kernels, question data
  movement (dtypes, layouts, allocations), and invent new hypotheses. Record them in KNOWLEDGE.md.
- Numerics are not a lever you pull: under `strict` (the default) every kernel reproduces the reference
  outputs (fp32 accumulation, exact argmins via coarse pass + exact re-rank). Only a human may set a
  different policy in `GOAL.md`. Never distillation, retraining or fine-tuning — the outputs stay the
  frozen reference model's.

## Backends — CUDA C++, and why

**Every kernel you write here is hand-written CUDA C++** (`fastkernel.backends.cuda_cpp.load_cuda_inline`),
captured with CUDA graphs. Triton, TileLang and CuTe are not used. This is a measured policy, not a taste:

- The wins left in a mature campaign are **fusion granularity** — a whole SEANet resnet block or a whole
  RVQ stage in *one* launch, so its intermediates never reach global memory. Adopting a CUDA-fused lineage
  of `campaigns/mimi` was worth **+13.8 % in one experiment**, on a tree where a season of tile tuning had
  already reached 70–98 % of every measured floor and every further fusion measured as a loss.
- A tile DSL's automatic pipeline will not give you that control. `cp.async` is a DMA: it **cannot transform
  a value in flight**, so folding an activation into a staged load forces a barrier inside the very pipeline
  that exists to hide latency — measured at **3.6× slower**, at the theoretical best configuration.
- The corollary is that individually-measured fusions can each look like losses while their **joint** optimum
  wins. Three engineers measured three such fusions at +7.85…+73.32 %, 11.4× and 3.6× against a DSL
  incumbent; the build that does all three is 12.8 % faster, because the buffers they remove never exist
  there. **A hill-climb from a partially-fused tree cannot reach a fully-fused one.**

The skills that document how: `/cuda-cpp-kernels` (the implementation backend), `/cuda-graphs` (capture),
`/hub-kernels` (a pre-built CUDA kernel is still CUDA), `/numerical-verification` (keeping a kernel
bit-faithful to the reference). Leaving an op on stock torch is always a legitimate answer.

A missing toolchain is not a limit: `fast-kernel toolchain install --cuda 13.3` provides a self-contained
nvcc/CCCL/NVVM from pip wheels, and `ensure_cuda_home()` wires CUDA_HOME/PATH.

## Roles (multi-agent)

One agent *can* run the whole loop, but a serial loop is usually the wrong choice: experiments on
different targets are independent, each one costs minutes of build + gates + benchmark, and the
ranking only changes when something is accepted. **Once `PLAN.md` lists two or more targets with
real headroom, run them in parallel** — `fast-kernel auto --agents 3 --islands 2`, or delegate one
`fk-kernel-engineer` per target — instead of walking the list one experiment at a time. Reserve the
serial loop for when a single target dominates or the next step depends on the last result. Parallel
agents think and build concurrently; the *measurements* never overlap — every `fast-kernel eval` /
`baseline` / `profile` / `probe` holds a machine-wide GPU lock for its harness run, and the wait is not
charged to the experiment's timeout (explore in parallel, measure serially, enforced).

Delegate to the project subagents:
`fk-profiler` (hotspot analysis), `fk-kernel-engineer` (writes kernels), `fk-verifier` (debugs gate
failures), `fk-benchmarker` (noise, protocol questions), `fk-reviewer` (reads the diff before eval:
simplicity, API, hidden CPU syncs), `fk-librarian` (docs, prior experiments, templates). Parallel
workers on git worktrees: `fast-kernel auto --agents N` (proposals are re-measured on the incumbent).
Add `--islands K` to split the workers into K populations, each exploring a different band of the ranked
targets so the search does not collapse onto one hotspot (an island model). `fast-kernel beam` shows the
top-k accepted candidates — the search population, not just the single incumbent; use it to see whether
the campaign has diverse strong points to build on or has narrowed to one lineage.

## Logging

- `results.tsv` — one line per experiment (exp, commit, status, metric, value, speedup, VRAM, gates, description).
- `experiments/NNNN-slug/` — metrics.json, gates.json, profile.json, run.log, patch.diff, notes.md.
- `KNOWLEDGE.md` — auto experiment log + your insights (`fast-kernel note`). Read it every iteration.
- `.fast-kernel/memory.jsonl` — one structured **reflexion** per experiment (target, techniques, measured
  delta, verdict, failure class). Query it with `fast-kernel memory --target <id>`: it returns the measured
  outcomes of similar targets (what worked) and the *repair chain* (failures already seen — do not repeat).
  This is measured history, not advice; the technique you try next is still yours to choose.
- The dashboard (`fast-kernel dashboard`) and `fast-kernel report` (static HTML) show all of it.

## Never stop

The loop runs autonomously until the optimization is **exhausted** — a measured convergence signal:
`CONVERGE_AFTER` consecutive experiments that ran but improved nothing. At that point the loop stops
itself (the `loop.active` flag is cleared). Until then it keeps profiling, hypothesising, evaluating and
learning. It never asks the human whether to continue, and it never asks the human to run a stop command
— "Is this a good stopping point?" is not a question this program asks. A human may still stop it early
with `fast-kernel stop` / `fast-kernel loop stop`, but the program never solicits that.
