#!/usr/bin/env python3
"""PreToolUse hook: block Edit/Write/MultiEdit on protected campaign files (and on the harness while a loop runs)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import LIBRARY_PROTECTED, PROTECTED, campaign_of, find_campaigns, project_dir, read_input  # noqa: E402


def main() -> int:
    data = read_input()
    tool_input = data.get("tool_input") or {}
    raw = tool_input.get("file_path") or tool_input.get("path") or ""
    if not raw:
        return 0
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_dir() / path
    path = path.resolve()
    campaign = campaign_of(path)
    if campaign is not None:
        try:
            rel = path.relative_to(campaign)
        except ValueError:
            rel = None
        if rel is not None and rel.parts and (rel.parts[0] in PROTECTED or rel.name in PROTECTED):
            print(f"fast-kernel: `{rel}` is protected (harness-owned). Edit only files under candidate/. "
                  f"Write disagreements to KNOWLEDGE.md with `fast-kernel note`.", file=sys.stderr)
            return 2
    root = project_dir()
    try:
        rel_root = path.relative_to(root).as_posix()
    except ValueError:
        return 0
    if any(rel_root.startswith(p) for p in LIBRARY_PROTECTED):
        active = [c for c in find_campaigns(root) if (c / ".fast-kernel" / "loop.active").exists()]
        if active:
            print(f"fast-kernel: `{rel_root}` is part of the evaluation harness and a campaign loop is active "
                  f"({active[0].name}). Optimizations go under candidate/; the harness stays immutable during a campaign.",
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
