---
name: fk-experiment
description: Run exactly ONE fast-kernel experiment (profile -> one hypothesis -> edit candidate/ -> fast-kernel eval -> note) in the current campaign and stop. Designed for `/loop /fk-experiment` and for the Stop-hook loop. Use for "run one experiment", "next iteration", "try one more idea".
argument-hint: [campaign-dir] [--target <id>] [--technique <id>]
allowed-tools: Bash(fast-kernel *), Bash(fk *), Bash(uv run *), Bash(python *), Bash(.venv/bin/python *), Bash(git diff *), Bash(git log *), Bash(git status *), Bash(nvidia-smi *), Read, Edit, Write, MultiEdit, Glob, Grep
---

# /fk-experiment — one iteration of the research loop

Campaign: `$ARGUMENTS` if given, else the campaign found from the current directory (`fast-kernel status`).

1. **Read the state**: `fast-kernel status --brief`, `fast-kernel ideas`, `fast-kernel history -n 5`,
   PLAN.md (re-generate with `fast-kernel profile` if it predates the incumbent), KNOWLEDGE.md insights.
2. **One hypothesis**: the untried target × technique pair with the largest Amdahl gain, unless a focus
   (`--target/--technique`) was given. Write it as one line: "<what> for <target> via <technique/backend>;
   expect <x>% end-to-end".
3. **Implement** under `candidate/` only. Reuse `fast-kernel templates` (Triton norm/silu*mul/matmul/
   causal-dwconv/codebook-argmin, CUDA C++, TileLang GEMM, CuTe elementwise) and
   `fastkernel.backends.graphs.Graphed`. Add `report()` evidence. Keep the diff focused.
4. **Evaluate**: `fast-kernel eval -m "<hypothesis>" --technique <ids> --target <id>` (`--simpler` when the
   change deletes code at equal speed). On a trivial crash (typo/import/shape) fix once and re-run.
5. **Learn**: `fast-kernel note "<insight with numbers>" --tags <ids>`; if the result contradicts
   PLAN.md, say what changed.
6. **Stop** after one recorded experiment with a 3-line summary: `#N [status] metric (Δ%) speedup vs
   baseline; kernels; next idea`. Never ask whether to continue — the loop (`/loop`, Stop hook, or
   `fast-kernel auto`) calls this skill again.
