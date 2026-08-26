---
name: fk-profile
description: Profile the current candidate of a fast-kernel campaign, rank hotspots by Amdahl gain, and explain where time goes (launch bound vs compute vs memory). Use for "where is the time going", "profile the model", "what should we optimize next".
argument-hint: [campaign-dir] [--workload name]
allowed-tools: Bash(fast-kernel *), Bash(uv run *), Read, Glob, Grep
---

Run `fast-kernel profile` (optionally `--workload <name>`) in the campaign, then read `PLAN.md` and
`hotspots.json`. Summarise for the user / the loop:

- wall time vs GPU-busy time and kernel launches per call → is it launch/overhead bound?
- the top 5 targets: class, category, boundness, share, expected speedup, Amdahl gain, example shapes;
- for the #1 target, the first untried technique, its backend and skill;
- anything surprising vs the previous profile (`experiments/NNNN-*/profile.json`).

Use `/hotspot-analysis` for the reasoning rules. Do not edit code here.
