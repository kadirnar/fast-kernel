from __future__ import annotations

from pathlib import Path


def iteration_prompt(campaign_root: Path, *, target: str | None = None, technique: str | None = None, worker: str | None = None,
                     iteration: int | None = None, memory_note: str = "") -> str:
    focus = ""
    if target or technique:
        focus = ("\nFocus for this iteration: " + (f"target `{target}` " if target else "") + (f"technique `{technique}` " if technique else "")
                 + "(from PLAN.md / `fast-kernel ideas`). If it is clearly exhausted, pick the next best untried idea.")
    if memory_note:
        focus += ("\n\n" + memory_note + "\nThis is measured history, not advice: use it to avoid repeating a failed edit "
                  "and to build on what already worked; the technique you try next is still yours to choose.")
    who = f" You are worker `{worker}`; the campaign directory below is your private worktree." if worker else ""
    return f"""Run exactly ONE fast-kernel experiment in the campaign at `{campaign_root}` and then stop.{who}
{f'This is loop iteration {iteration}.' if iteration is not None else ''}
Follow AGENTS.md (the research program) strictly:
1. `cd {campaign_root}` then read `fast-kernel brief`: incumbent and threshold, the plateau streak, the ranked targets
   with their measured memory (what was tried, what worked, what failed), the last experiments and the latest insights.
   Read PLAN.md (shapes, top kernels), KNOWLEDGE.md and `fast-kernel memory --target <id>` only where the brief
   points you. Never repeat an identical failed edit.
2. Pick ONE target -- the one with the most measured headroom (biggest share of GPU time / most launches).
   The technique is yours to discover from the profile, the top kernels and KNOWLEDGE.md; nothing tells you
   which method to use or how much it will speed up. Never repeat an identical failed edit.
   THE IMPLEMENTATION BACKEND IS CUDA C++ (`/cuda-cpp-kernels`), with `/cuda-graphs` for capture and stock
   torch ops where they already win. Do not write Triton, TileLang or CuTe kernels. This is a measured
   policy, not a preference: what is left in a mature campaign is fusion granularity -- a whole block or a
   whole quantizer stage in ONE launch so its intermediates never reach global memory -- and a tile DSL's
   automatic pipeline will not give you that. `cp.async` is a DMA and cannot transform a value in flight, so
   an activation function folded into a staged load costs a barrier inside the pipeline that exists to hide
   latency; that was measured at 3.6x. Rank by ABSOLUTE excess over a floor computed from shapes you read out
   of the module, never by ratio, and check any two-term fit's intercept against an empty kernel at the same
   grid before believing it.
3. Implement it ONLY under `candidate/` (candidate/__init__.py `apply(model, ctx)` and candidate/kernels/*),
   in CUDA C++ via `fastkernel.backends.cuda_cpp.load_cuda_inline`. Never edit
   GOAL.md, spec.py, experiments/, results.tsv or anything under the fastkernel package. If a kernel needs a library that
   is not installed, install it (`uv pip install ...`) and continue -- never stop to ask a human for it.
4. Run `fast-kernel eval -m "<one-line hypothesis>" --technique <ids> --target <id>` (add `--simpler` only when the change
   deletes code at equal speed). If it crashes on a trivial error (typo, import, shape), fix and re-run once; otherwise accept
   the revert and move on.
5. Read the verdict, then record one insight: `fast-kernel note "<what you learned, with numbers>" --tags <ids>`.
Do not ask questions, do not pause for confirmation, do not stop early to summarise, and never ask the human whether to
continue or to stop; the harness decides keep/revert and the loop runs until the optimization is exhausted.{focus}
"""


def campaign_prompt(campaign_root: Path) -> str:
    return f"""Continuously optimize the model in the fast-kernel campaign at `{campaign_root}`: repeat the
one-experiment procedure from AGENTS.md indefinitely (profile -> hypothesis -> edit candidate/ -> `fast-kernel eval` ->
learn -> next). Each experiment builds on the accepted incumbent. Do not stop, do not ask whether to continue; if you are
interrupted, the campaign resumes from its persisted state with `fast-kernel status`."""
