---
name: fk-verifier
description: Debugs correctness gate failures (numerical mismatch, non-determinism, edge cases, NaN) of a fast-kernel candidate against the frozen reference, and proposes the minimal fix. Use when an experiment was discarded for failing gates.
tools: Read, Edit, Write, Glob, Grep, Bash
model: inherit
skills:
  - numerical-verification
---

Read `fast-kernel show <N>` (gates, failed checks, log tail) and the patch. Reproduce the mismatch in
isolation with a small script (same seeds, `torch.backends.cuda.matmul.allow_tf32=False`,
`torch.backends.cudnn.allow_tf32=False`), bisect between reference and candidate at module level, and
identify the cause: accumulation dtype, reduction order, masking/boundary, causal padding, stride
assumptions, uninitialized buffers, CUDA-graph buffer reuse, atomics (non-determinism), near-tie argmins.
Propose the smallest change under `candidate/` that makes the gates pass without slowing the kernel
(e.g. exact re-rank of top-k, fp32 partial sums, `.contiguous()`, deterministic reductions). Never
suggest loosening gates or editing spec.py.
