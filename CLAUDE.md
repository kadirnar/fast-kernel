@AGENTS.md

## Claude Code specifics for this repository

- Commands: `uv sync --extra cuda` (GPU runtime), `uv run fast-kernel doctor`, `uv run pytest -q`.
- Campaigns live under `campaigns/<name>/` (gitignored, each is its own git repo). Start one with
  `fast-kernel init mimi|lfm25|lfm-audio|yolo|custom`; the skill `/fk-optimize <model-or-dir>` does
  init → probe → baseline → dashboard → loop in one go.
- Skills: `/fk-optimize`, `/fk-experiment` (one iteration; use with `/loop /fk-experiment`),
  `/fk-profile`, `/fk-verify`, `/fk-bench`, `/fk-status`, `/fk-dashboard`, `/fk-report`,
  `/fk-parallel`, `/fk-add-model`, and the backend skills `/triton-kernels`, `/tilelang-kernels`,
  `/cute-dsl-kernels`, `/cuda-cpp-kernels`, `/cuda-graphs`, `/torch-compile`, `/hub-kernels`,
  `/numerical-verification`.
- The Stop hook keeps a session iterating while a campaign's `.fast-kernel/loop.active` flag exists
  (`fast-kernel loop start|stop`). The PreToolUse hook blocks edits to protected campaign files.
- Library development (not campaign work): edit `src/fastkernel/`, run `uv run pytest -q`; keep the
  core stdlib-only and import torch lazily.
