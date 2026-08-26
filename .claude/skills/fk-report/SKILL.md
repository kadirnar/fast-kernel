---
name: fk-report
description: Export a self-contained HTML report of a fast-kernel campaign (same graphs as the dashboard, data embedded) and summarise the results. Use for "write up the results", "export the experiments", "share the graph".
argument-hint: [campaign-dir] [--out report.html]
allowed-tools: Bash(fast-kernel *), Bash(uv run *), Read
---

Run `fast-kernel report [--out ...]` in the campaign and then summarise in prose: baseline vs incumbent
(metric, speedup, rtf/tokens-per-second/fps, kernel launches), the accepted experiments in order with
their individual gains (the lineage), the most informative failures from KNOWLEDGE.md, and the untried
ideas with the largest expected gain. Point to `results.tsv` and the report path.
