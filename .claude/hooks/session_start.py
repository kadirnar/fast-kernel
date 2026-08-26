#!/usr/bin/env python3
"""SessionStart hook: surface the campaigns in this project and their state as context."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import experiment_count, find_campaigns, last_experiment, project_dir  # noqa: E402


def main() -> int:
    root = project_dir()
    campaigns = find_campaigns(root)
    lines = ["fast-kernel: autoresearch harness for model inference (read AGENTS.md)."]
    if not campaigns:
        lines.append('No campaign yet. The user starts one with plain text, e.g. "Optimize the Mimi codec model." (-> fk-optimize skill).')
    for c in campaigns:
        flags = [f for f in ("loop.active", "paused", "stop") if (c / ".fast-kernel" / f).exists()]
        last = last_experiment(c) or {}
        lines.append(f"- campaign {c.relative_to(root) if c.is_relative_to(root) else c}: {experiment_count(c)} experiments"
                     + (f", last #{last.get('number')} [{last.get('status')}] {str(last.get('description', ''))[:60]}" if last else "")
                     + (f", flags: {', '.join(flags)}" if flags else ""))
    lines.append('Plain text drives everything: "Optimize the <model>." starts/continues the loop (fk-optimize), "Stop optimizing." ends it. Live graph: `fast-kernel dashboard`.')
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
