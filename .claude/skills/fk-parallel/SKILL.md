---
name: fk-parallel
description: Run several fast-kernel agents at once on one campaign - in-session engineers in private worktrees whose proposals are measured serially by `fast-kernel inbox`, or headless workers (`fast-kernel auto --agents N`). Use for "use multiple agents", "parallelize the search", "run 3 engineers".
argument-hint: [--agents N] [--targets t_a,t_b,...]
allowed-tools: Bash(fast-kernel *), Bash(uv run *), Bash(git *), Read, Agent
---

# /fk-parallel — one round with N agents

The GPU and the incumbent are shared, so the protocol is: **explore in parallel, measure serially** — and
the harness enforces the second half: `fast-kernel eval` / `baseline` / `profile` / `probe` take a machine-wide
GPU lock for the harness run (`gpu.waited` events record the wait; it is not charged to the timeout).

## In-session (Agent tool)

1. `uv run fast-kernel ideas` → pick N distinct targets (most measured headroom, one approach each).
2. For each target: `uv run fast-kernel worktree create eng-<target>` (prints an absolute path; a private
   git worktree branched from the incumbent with PLAN.md/KNOWLEDGE.md/incumbent copied in).
3. Spawn N `fk-kernel-engineer` subagents in parallel. Each prompt contains: the worktree path, the
   target id + class + shapes from PLAN.md, the technique and backend, and
   the instruction to finish with `cd <worktree> && uv run fast-kernel propose -m "..." --technique <id> --target <id>`.
   Engineers self-test on the GPU (small scripts) but do not run `fast-kernel eval` in parallel.
4. When they return: `cd <campaign> && uv run fast-kernel inbox` — applies each proposal on the current
   incumbent (`git apply --check`), runs the full harness, keeps or reverts, moves the files to
   `.fast-kernel/inbox/processed|failed/`. When the incumbent moved under a proposal, the inbox first tries a
   3-way merge from the recorded blob ids (worktrees share the object store), so only genuinely overlapping
   changes are rejected — with an event and the commit to rebase on; ask that engineer to rebase
   (`git checkout -B worker/<name> <incumbent>`) and re-propose.
5. `fast-kernel note` the round's lessons; `fast-kernel worktree remove eng-<target>` for finished ones;
   re-rank (`fast-kernel profile` if a keep changed the picture); next round.

Use `fk-verifier` on a proposal that failed gates, `fk-reviewer` before applying an expensive one,
`fk-librarian` for references. Roles never grade their own work.

## Headless

`uv run fast-kernel auto --agents N [--iterations K]` does the same with `claude -p` workers: worktrees
under `.fast-kernel/worktrees/wN`, hotspot leases so two workers never take the same target, proposals in
`.fast-kernel/inbox/`, serial promotion by the main loop. Watch the dashboard's agents panel;
`fast-kernel stop` ends everything after the current experiment.
