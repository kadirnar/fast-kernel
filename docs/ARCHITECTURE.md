# Architecture

fast-kernel is three things that share one directory layout: a **fixed evaluation harness** (the
part an agent must never touch), an **agent program** (AGENTS.md + Claude Code agents/skills/hooks that
run the loop), and a **live record** (SQLite events, results.tsv, the dashboard).

```
src/fastkernel/
  campaign.py        campaign directory, git lineage (branch fast-kernel/<model>, tags exp-N), flags
  config.py          GOAL.md frontmatter -> GoalConfig (metric, direction, thresholds, gate policy, bench)
  store.py           SQLite/WAL: experiments, events (SSE source), agents, leases, kv
  results.py         results.tsv ledger
  knowledge.py       KNOWLEDGE.md (auto experiment log + agent insights)
  playbook.py        technique catalogue (tiers 0-6), internal only — never rendered to the agent
  memory.py          campaign memory: per-experiment reflexion, repair chains, failure classification
  models/            ModelSpec contract + built-ins (mimi, lfm25, lfm-audio, yolo, hf-causal-lm, torch module)
  profiling/         trace (torch.profiler + module hooks + Python frames) -> classify (roofline) -> rank (measured headroom) -> plan
  harness/           gates (5 stages) -> bench (fixed protocol) -> run.py (subprocess) -> evaluate.py (keep/revert)
  backends/          probes + helpers: triton, tilelang, cute-dsl, cuda-cpp (nvcc discovery), torch.compile, cuda-graphs,
                     hub-kernels, toolchain (pip-wheel CUDA toolchains); templates/ starter kernels
  dashboard/         stdlib HTTP + SSE server, static single-page app, self-contained report export
  agents/            headless driver (claude -p per experiment), worktree workers, inbox promotion, prompts
  templates/models/  what `fast-kernel init <model>` copies (GOAL.md, candidate/, PROMPT.md, notes)
  cli.py             fast-kernel / fk
```

## Trust boundary

The harness owns: loading the frozen reference, generating seeded inputs, running the candidate, the
five correctness gates, the benchmark protocol, the profiler, and the keep/revert decision with git.
The agent owns exactly one tree: `candidate/`. A PreToolUse hook blocks edits to protected paths; the
harness re-reads GOAL.md and spec.py from disk every run and never trusts the candidate's own claims
(`report()` is evidence shown to humans, not an input to the decision).

Commands are argv lists, run in a subprocess with a timeout and a process group so a hung kernel
cannot hang the loop. Experiment artifacts are written before any git state changes.

## State machines

Experiment: `running -> baseline | keep | discard | crash | error`.
`keep`/`baseline` commit `candidate/` and move the incumbent; `discard`/`crash` reset `candidate/` to
the incumbent (patch kept under `experiments/`); `error` (harness/reference failure) leaves the tree
untouched so the environment can be repaired.

Loop (Claude Code): `loop.active` flag -> Stop hook blocks the end of turn with "run the next
experiment" while new experiments keep appearing; `paused` -> flag honoured by drivers; `stop` ->
everything winds down after the current experiment. The loop also ends itself on measured convergence
(15 consecutive experiments that ran but improved nothing): the driver breaks and the hook clears
`loop.active`. Every finished experiment appends one reflexion to `.fast-kernel/memory.jsonl`
(target, techniques, measured delta, verdict, failure class), queried with `fast-kernel memory`.

## Data flow of one experiment

1. `fast-kernel eval -m "..."` snapshots the candidate diff vs the incumbent (git), allocates
   `experiments/NNNN-slug/`, records `experiment.started`.
2. `python -m fastkernel.harness.run` (subprocess): load spec -> reference model -> seeded inputs ->
   reference outputs (and noise floor on baseline) -> free reference -> `candidate.apply` -> gates ->
   bench -> profile (+ targets ranked by measured headroom, with the matrix from history) -> JSON files.
3. Decision + git + incumbent + results.tsv + KNOWLEDGE.md + events; PLAN.md regenerated on keep.
4. The dashboard tails the events table over SSE and re-renders.

## Why a whole-model harness rather than KernelBench-style single kernels

Inference speed at small batch is dominated by launch count and Python overhead, and the biggest
wins (CUDA graphs, fusion across modules) are invisible when a single op is benchmarked in isolation.
fast-kernel measures the model's real workload, attributes GPU time to modules *and* to non-forward
methods (quantizer `encode`, `generate`), and lets the agent change anything under `candidate/` —
kernels, module swaps, graph capture — while the gates hold the end-to-end contract fixed.
