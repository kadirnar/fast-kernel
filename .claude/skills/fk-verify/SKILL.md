---
name: fk-verify
description: Diagnose a failed correctness gate (numerical, determinism, shapes, edge) of the latest or a given fast-kernel experiment and propose the minimal fix. Use for "why did the gates fail", "mismatch vs reference", "non-deterministic".
argument-hint: [experiment-number]
allowed-tools: Bash(fast-kernel *), Bash(uv run *), Bash(python *), Bash(.venv/bin/python *), Read, Glob, Grep, Edit, Write
---

1. `fast-kernel show <N> --log` (default: latest) → failed checks with values vs thresholds, log tail.
2. `fast-kernel show <N> --patch` → what changed.
3. Reproduce in isolation with `uv run python` in the campaign dir: load the spec
   (`from fastkernel.models.spec import load_spec`), build reference + candidate, run the failing workload,
   and bisect module by module (`torch.testing.assert_close`, max-abs / rel error, argmax agreement).
4. Follow `/numerical-verification` for the usual causes and exact-preserving fixes.
5. Apply the fix under `candidate/` and re-run `fast-kernel eval -m "fix: ..." --technique <ids> --target <id>`.
Never propose editing spec.py or loosening GOAL.md.
