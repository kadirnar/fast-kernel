---
name: fk-bench
description: Re-measure or sanity-check fast-kernel timings (noise floor, repeats, contention, clocks) without changing code. Use for "is this speedup real", "results are noisy", "re-run the benchmark".
argument-hint: [--repeats N]
allowed-tools: Bash(fast-kernel *), Bash(uv run *), Bash(nvidia-smi *), Read
---

- `nvidia-smi` → other processes / utilisation / clocks / temperature (contention inflates noise).
- `fast-kernel status` → the noise floor measured at baseline and the effective acceptance threshold.
- `fast-kernel eval --force -m "re-measure" --repeats 200` re-measures the *current* candidate as a new
  experiment (kept only if it beats the incumbent by the threshold — it should not, so it lands as discard
  with the numbers you need); `fast-kernel baseline --force` re-baselines when the machine state changed.
- Compare median vs min vs p90 in `fast-kernel show <N>`; a big median/min gap means contention or clocks.
- Confirm the fast path really ran: `candidate_report` in `fast-kernel show <N> --json`.
Never hand-edit results.tsv or experiment records.
