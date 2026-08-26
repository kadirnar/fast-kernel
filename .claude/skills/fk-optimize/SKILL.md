---
name: fk-optimize
description: Start or continue an endless fast-kernel optimization campaign for a model (mimi, lfm25, lfm-audio, yolo, a campaign directory, or any torch module). Use when the user says "optimize <model>", "make <model> faster", "accelerate the Mimi codec", "continue the campaign", or "run kernel experiments".
argument-hint: <model-or-campaign-dir> [--precision strict|tolerant] [--set key=value]
allowed-tools: Bash(fast-kernel *), Bash(fk *), Bash(uv *), Bash(python *), Bash(.venv/bin/python *), Bash(git diff *), Bash(git log *), Bash(git status *), Bash(nvidia-smi *), Read, Edit, Write, MultiEdit, Glob, Grep
---

# /fk-optimize — the endless loop, from a single line of text

Arguments: `$ARGUMENTS` (a built-in model name — `mimi`, `lfm25`, `lfm-audio`, `yolo`, `custom` — or an
existing campaign directory; optional `--precision` / `--set key=value` overrides for GOAL.md).

## 1. Prepare (idempotent)

```bash
uv run fast-kernel doctor                      # torch/CUDA/backends/claude present? fix hints if not
uv run fast-kernel init <model> [--precision ..] [--set ..]   # skip if the campaign dir already exists
cd campaigns/<model>
uv run fast-kernel probe                       # GPU + roofline + backend compile probes -> capabilities.json
uv run fast-kernel baseline                    # experiment #0 + noise floor + PLAN.md (skip if it exists)
uv run fast-kernel loop start                  # Stop hook keeps this session iterating
```

Start the live graph in the background if it is not running: `uv run fast-kernel dashboard --root campaigns`
(http://127.0.0.1:8765) and tell the user the URL once.

If `uv sync --extra cuda` has not been run, run it. If a model extra is missing (`--extra yolo`,
`--extra audio`, `--extra tilelang`, `--extra cute`, `--extra hub`), install it. When `probe` shows
`tilelang` or `cuda-cpp` not READY with a host-compiler error (gcc newer than nvcc supports, `cudafe++`
crash, missing `nvcc`), run `uv run fast-kernel toolchain install --cuda 13.3` and probe again — never
treat a toolchain gap as a hardware limit.

## 2. Loop (never stops on its own)

Repeat the `/fk-experiment` procedure forever: status → ideas → one hypothesis → edit `candidate/` →
`fast-kernel eval` → `fast-kernel note` → next. Follow AGENTS.md for rules, decision logic, crash protocol
and plateau strategy. Delegate to `fk-kernel-engineer`, `fk-verifier`, `fk-profiler` subagents when it
helps; use `/fk-parallel` for several workers.

Report progress to the user only as short interleaved lines (experiment number, status, metric, speedup);
never end your turn with a question. The loop ends when the user says so (`fast-kernel loop stop`).
