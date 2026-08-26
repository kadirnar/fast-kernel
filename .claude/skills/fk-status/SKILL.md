---
name: fk-status
description: Show the state of the fast-kernel campaign(s): incumbent, speedup vs baseline, last experiments, loop flags, top targets. Use for "how is the campaign going", "status", "progress".
argument-hint: [campaign-dir]
allowed-tools: Bash(fast-kernel *), Bash(uv run *), Read
---

Run `fast-kernel status` and `fast-kernel history -n 10` in the campaign (`$ARGUMENTS` or auto-detected)
and report: experiments (keep/discard/crash counts), incumbent value vs baseline (speedup), kernel
launches then vs now, the last three experiments with reasons, loop state, and the next best idea from
`fast-kernel ideas`. Mention the dashboard URL if a server is running (`curl -s http://127.0.0.1:8765/api/campaigns`).
