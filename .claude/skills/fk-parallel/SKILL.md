---
name: fk-parallel
description: Run several fast-kernel agents in parallel (git worktrees + hotspot leases + inbox promotion) or delegate one iteration to the project subagents. Use for "use multiple agents", "parallelize the search", "run 3 workers".
argument-hint: [--agents N] [--iterations K]
allowed-tools: Bash(fast-kernel *), Bash(uv run *), Read, Agent
---

Two modes:

1. **Headless workers** (recommended for long runs): `uv run fast-kernel auto --agents N [--iterations K]`
   in the campaign. Each worker gets its own git worktree (`.fast-kernel/worktrees/wN`) and leases one
   hotspot target; its accepted patches land in `.fast-kernel/inbox/` and the main loop re-evaluates them
   on top of the incumbent (serial, so the lineage stays one measured chain). Watch the dashboard's agents
   panel; `fast-kernel inbox` processes proposals manually; `fast-kernel stop` ends everything.
2. **In-session subagents**: spawn `fk-profiler`, then several `fk-kernel-engineer` agents (one per
   target, `isolation: worktree` is set for engineers when parallel), then evaluate their patches one by
   one with `fast-kernel eval` (apply each with `git apply` in the campaign). Use `fk-reviewer` before
   expensive evals and `fk-verifier` on gate failures.

Roles are separated so no agent grades its own work: the harness measures, the reviewer reads, the
engineer writes.
