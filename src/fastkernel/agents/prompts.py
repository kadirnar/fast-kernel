from __future__ import annotations

from pathlib import Path


def iteration_prompt(campaign_root: Path, *, target: str | None = None, technique: str | None = None, worker: str | None = None,
                     iteration: int | None = None) -> str:
    focus = ""
    if target or technique:
        focus = ("\nFocus for this iteration: " + (f"target `{target}` " if target else "") + (f"technique `{technique}` " if technique else "")
                 + "(from PLAN.md / `fast-kernel ideas`). If it is clearly exhausted, pick the next best untried idea.")
    who = f" You are worker `{worker}`; the campaign directory below is your private worktree." if worker else ""
    return f"""Run exactly ONE fast-kernel experiment in the campaign at `{campaign_root}` and then stop.{who}
{f'This is loop iteration {iteration}.' if iteration is not None else ''}
Follow AGENTS.md (the research program) strictly:
1. `cd {campaign_root}` then read `fast-kernel status --brief`, `fast-kernel ideas`, PLAN.md, KNOWLEDGE.md and the last
   3 experiments (`fast-kernel history -n 3`). Never repeat an identical failed edit.
2. Choose ONE hypothesis with the largest expected end-to-end (Amdahl) gain among untried target x technique pairs.
3. Implement it ONLY under `candidate/` (candidate/__init__.py `apply(model, ctx)` and candidate/kernels/*). Never edit
   GOAL.md, spec.py, experiments/, results.tsv or anything under the fastkernel package.
4. Run `fast-kernel eval -m "<one-line hypothesis>" --technique <ids> --target <id>` (add `--simpler` only when the change
   deletes code at equal speed). If it crashes on a trivial error (typo, import, shape), fix and re-run once; otherwise accept
   the revert and move on.
5. Read the verdict, then record one insight: `fast-kernel note "<what you learned, with numbers>" --tags <ids>`.
Do not ask questions, do not pause for confirmation, do not stop early to summarise; the harness decides keep/revert.{focus}
"""


def campaign_prompt(campaign_root: Path) -> str:
    return f"""Continuously optimize the model in the fast-kernel campaign at `{campaign_root}`: repeat the
one-experiment procedure from AGENTS.md indefinitely (profile -> hypothesis -> edit candidate/ -> `fast-kernel eval` ->
learn -> next). Each experiment builds on the accepted incumbent. Do not stop, do not ask whether to continue; if you are
interrupted, the campaign resumes from its persisted state with `fast-kernel status`."""
