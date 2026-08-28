---
name: fk-experiment
description: Run exactly ONE fast-kernel experiment (state -> one hypothesis -> edit candidate/ -> fast-kernel eval -> note) in the current campaign and stop. The unit of work for /fk-optimize, `/loop /fk-experiment` and the Stop-hook loop. Use for "run one experiment", "next iteration", "try one more idea".
argument-hint: [campaign-dir] [--target <id>] [--technique <id>]
allowed-tools: Bash(fast-kernel *), Bash(fk *), Bash(uv run *), Bash(python *), Bash(.venv/bin/python *), Bash(git diff *), Bash(git log *), Bash(git status *), Bash(nvidia-smi *), Bash(cat *), Bash(ls *), Read, Edit, Write, MultiEdit, Glob, Grep
---

# /fk-experiment — one iteration, in full

Campaign: `$ARGUMENTS` if a directory was given, else `uv run fast-kernel resolve "continue" --json` →
`campaign`. Run every command as `cd <campaign> && uv run fast-kernel ...`.

## 1. Read the state (2 minutes, no shortcuts)

```bash
uv run fast-kernel status --brief          # incumbent, speedup, threshold, loop state
uv run fast-kernel ideas                    # hotspots by measured headroom (share x (1 - SOL)), with what was tried
uv run fast-kernel history -n 5             # what just happened and why
```
Then read `PLAN.md` (regenerate with `uv run fast-kernel profile` if it predates the incumbent),
`KNOWLEDGE.md` (insights first) and, if the
last experiment failed, `uv run fast-kernel show <N> --log`.

## 2. Choose one hypothesis

Score = share × (1 − roofline efficiency), from `ideas`. Tie-breaks: untried > larger measured share >
lower SOL > smaller diff. The technique itself is never prescribed — discover it from the measurements. A focus given as `--target/--technique` wins unless it is
clearly exhausted. Write the hypothesis as one line before touching code:
`"<what> for <target class> via <technique/backend>; expect ~<x>% end-to-end because <share/boundness>"`.

Do not pick: an identical failed edit; a technique whose skill you have not read for this backend;
anything that changes GOAL.md's policy.

## 3. Implement (candidate/ only)

- Kernel in `candidate/kernels/<name>.py`, integration in `candidate/__init__.py: apply(model, ctx)`
  (module swap or forward monkeypatch; keep signatures), `report()` evidence (`active`, `invocations`).
- Start from `uv run fast-kernel templates` (Triton rmsnorm / silu*mul / autotuned GEMM with epilogue /
  causal depthwise conv1d / codebook argmin; CUDA C++; TileLang GEMM; CuTe elementwise) and
  `fastkernel.backends.graphs.Graphed` / `ShapeBucketedGraphs`.
- Numerics per `/numerical-verification`: fp32 accumulation, fixed-order reductions, exact argmin via
  coarse pass + fp32 re-rank, reference tie-breaks and padding. Strict policy = identical outputs.
- Self-test before eval: a 20-line script in `/tmp` comparing the kernel with the torch reference on
  the real shapes from PLAN.md (`torch.testing.assert_close`, plus `torch.equal` for discrete outputs).
- Warm up every shape in `apply()`; autotune results cached under `candidate/tuned/`; no `.item()`,
  `.cpu()`, data-dependent Python or allocations in the hot path when graphs are involved.

## 4. Evaluate

```bash
uv run fast-kernel eval -m "<the one line>" --technique <id[,id]> --target <id>   # add --simpler only when deleting code at equal speed
```
Read the verdict: gates (which stage/check failed, value vs threshold), metric vs incumbent vs
threshold, kernel count, GPU busy, next target. Crash with a trivial cause (typo, import, stride,
`.contiguous()`, meta tensor) → fix and rerun once. Anything else → accept the revert.

The verdict line also names the **decision basis**. `anchored ratio vs the reference model` means the
candidate and the reference were timed interleaved in one process, so the comparison is free of
session-to-session drift and the threshold is that measurement's own uncertainty (typically well
under 1 %). `raw milliseconds` means no anchor was available and the threshold falls back to the
baseline noise floor — one `fast-kernel eval --force` on a clean tree fixes that for good.

**bank** = the gain was real but below the resolution limit. The harness committed it and left it in
`candidate/`; build the next experiment on top of it rather than reverting or re-testing it. The
incumbent deliberately does not move, so the accumulated tree is compared against the last number
the campaign can defend, and is promoted in one keep when the pile clears the threshold.

## 5. Learn and stop

```bash
uv run fast-kernel note "<insight with numbers: what, how much, why, what next>" --tags <technique,target>
```
Then stop after exactly one recorded experiment with three lines:
`#N [status] <metric> (Δ vs incumbent) · <speedup>× vs baseline · kernels <k>`,
`gates: ...` (or the failing check), `next: <the hypothesis you would run next>`.
Never ask whether to continue — the loop calls this skill again.
