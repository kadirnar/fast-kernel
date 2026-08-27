---
name: fk-loop
description: Explains and configures the three ways to keep fast-kernel iterating until the optimization is exhausted - `/loop /fk-experiment` (scheduled wake-ups), the Stop-hook loop (`fast-kernel loop start`), and the headless driver (`fast-kernel auto`). Use for "keep going overnight", "run this in a loop", "never stop".
argument-hint: [start|stop|status]
allowed-tools: Bash(fast-kernel *), Bash(uv run *), Read
---

- **`/loop /fk-experiment`** — Claude Code re-invokes the one-iteration skill on a self-paced or fixed
  interval (`/loop 10m /fk-experiment`). Survives `--resume`; stop with Esc / `/loop` controls.
- **Stop-hook loop** — `fast-kernel loop start` creates `.fast-kernel/loop.active`; the project's Stop hook
  then blocks every attempt to end the turn with "run the next experiment" while progress is being made
  (it lets the session end after 3 consecutive stops without a new experiment). It also ends the loop
  itself once the optimization is exhausted — 15 consecutive experiments that ran but improved nothing —
  by clearing the flag. `fast-kernel loop stop` ends it early. This is the most robust interactive mode.
- **Headless** — `uv run fast-kernel auto [--agents N] [--model ...]` runs `claude -p` once per experiment
  until the optimization is exhausted (survives session limits, logs agent activity to the dashboard);
  `--agents N [--islands K]` adds parallel worktree workers. `fast-kernel pause|resume|stop`.

`$ARGUMENTS` = start|stop|status → run `fast-kernel loop $ARGUMENTS` in the campaign and confirm.
