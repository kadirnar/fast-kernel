---
name: fk-optimize
description: Start or continue the fast-kernel optimization loop for a model, from plain text. Use whenever the user asks to optimize, accelerate, speed up or make faster a model - "Optimize the Mimi codec model.", "make LFM2.5 faster", "optimize the LFM2 audio model", "optimize the YOLO model", "optimize the PyTorch model in ./x.py", "continue the optimization".
argument-hint: <what the user said>
allowed-tools: Bash(fast-kernel *), Bash(fk *), Bash(uv *), Bash(python *), Bash(.venv/bin/python *), Bash(git diff *), Bash(git log *), Bash(git status *), Bash(nvidia-smi *), Bash(curl -s http://127.0.0.1:*), Read, Edit, Write, MultiEdit, Glob, Grep
---

# /fk-optimize — the endless loop, from one sentence

The user writes plain text (`$ARGUMENTS`). Map it to a model, never ask them to choose options:

| the user mentions | model name |
|---|---|
| Mimi, codec, kyutai | `mimi` |
| LFM2.5, LFM 2.5, Liquid | `lfm25` |
| LFM2 audio, LFM audio, speech-to-speech | `lfm-audio` |
| YOLO, detection | `yolo` |
| a file or directory path | `custom` — run the `/fk-add-model` procedure on that path first |
| an existing `campaigns/<name>` directory | continue that campaign |

Quality rule: the precision policy in GOAL.md stays `strict` (identical outputs) unless the user
explicitly asked for something else in their own words. Never pass `--precision`, never edit GOAL.md.

## 1. Prepare (idempotent; say what you are doing in one line each)

```bash
uv run fast-kernel doctor                     # torch/CUDA/backends present? apply its fix hints
uv run fast-kernel init <model>               # skip if campaigns/<model> already exists
cd campaigns/<model>
uv run fast-kernel probe                      # GPU + backend probes -> capabilities.json
uv run fast-kernel baseline                   # experiment #0 + noise floor + PLAN.md (skip if it exists)
uv run fast-kernel loop start                 # the Stop hook keeps this session iterating
```

Start the dashboard in the background if `curl -s http://127.0.0.1:8765/api/campaigns` does not answer:
`uv run fast-kernel dashboard --root campaigns` — then tell the user the printed URL once.

Missing pieces are fixed, never reported as limits: `uv sync --extra cuda` if torch is missing; the
model extras (`--extra yolo`, `--extra audio`, `--extra tilelang`, `--extra cute`, `--extra hub`); and
`uv run fast-kernel toolchain install --cuda 13.3` when `probe` shows `tilelang` or `cuda-cpp` failing
with a host-compiler error. Re-probe after every fix.

## 2. Loop (never stops on its own)

Repeat the `/fk-experiment` procedure: status → ideas → one hypothesis → edit `candidate/` only →
`fast-kernel eval` → `fast-kernel note` → next. Follow AGENTS.md (quality contract, rules, crash
protocol, plateau strategy). Delegate to `fk-kernel-engineer`, `fk-verifier`, `fk-profiler` subagents
when it helps.

Report progress as short lines (experiment number, kept/discarded, latency, speedup). Never end your
turn with a question. The loop ends only when the user says to stop ("Stop optimizing." →
`fast-kernel loop stop`).
