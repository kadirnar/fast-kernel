@AGENTS.md

## Claude Code specifics for this repository

- **Plain text is the interface.** When the user asks, in any words, to optimize / accelerate /
  speed up a model (e.g. "Optimize the Mimi codec model.", "make LFM2.5 faster", "optimize the model
  in ./x.py"), invoke the `fk-optimize` skill immediately — do not ask which mode or option they want.
  The skill's first command, `uv run fast-kernel resolve "<sentence>" --json`, maps the words to the
  model, to the absolute campaign folder (`campaigns/<model>`, reused if present) and to the remaining
  steps; every command then runs as `cd <campaign> && uv run fast-kernel ...`.
  "Stop optimizing." means `fast-kernel loop stop` (and `fast-kernel stop` if a headless driver runs).
- Commands: `uv sync --extra cuda` (GPU runtime), `uv run fast-kernel doctor`,
  `uv run --extra dev pytest -q` (pytest lives in the `dev` extra).
- Campaigns live under `campaigns/<name>/` (gitignored, each its own git repo).
- Skills: `fk-optimize` (the loop), `fk-experiment` (one iteration; also for `/loop /fk-experiment`),
  `fk-profile`, `fk-verify`, `fk-bench`, `fk-status`, `fk-dashboard`, `fk-report`, `fk-parallel`,
  `fk-add-model`, `fk-loop`, and the backend skills (`cuda-cpp-kernels` -- the implementation
  backend -- `cuda-graphs`, `hub-kernels`, `numerical-verification`).
- Beyond the loop commands: `fast-kernel brief` (one screen: state, plateau streak, ranked targets with their measured
  memory, last experiments, insights -- the first command of every iteration), `fast-kernel memory --target <id>`
  (measured history of a target), `fast-kernel beam` (top-k accepted candidates) and
  `fast-kernel auto --agents N --islands K`.
- The Stop hook keeps a session iterating while a campaign's `.fast-kernel/loop.active` flag exists;
  it clears the flag itself once the optimization is exhausted (15 experiments with no improvement).
  The PreToolUse hook blocks edits to protected campaign files.
- Library development (not campaign work): edit `src/fastkernel/`, run `uv run --extra dev pytest -q`;
  keep the core stdlib-only and import torch lazily.
