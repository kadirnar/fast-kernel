"""KNOWLEDGE.md: the campaign's accumulated learnings (the 'ideas that did / did not pay off' table).

Two sections: an auto-appended experiment log (one line per experiment with numbers) and free-form
insights added by the agent with `fast-kernel note`. The loop reads it before proposing the next
hypothesis so the search does not repeat itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import fmt, now_iso

HEADER = """# Knowledge base

Accumulated evidence for this campaign. Numbers beat priors: an idea that failed here is a fact about
*this* model on *this* machine, not a law. Re-measure before assuming, but do not repeat identical
failed edits.

## Insights

(add with `fast-kernel note "..." --tags fused-quantizer,rvq`)

## Experiment log

| # | status | metric | delta vs incumbent | kernels | techniques | target | description |
|--:|---|--:|--:|--:|---|---|---|
"""


def ensure(path: Path) -> None:
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")


def append_experiment(path: Path, record: dict[str, Any]) -> None:
    ensure(path)
    delta = record.get("improvement")
    line = (f"| {record.get('number')} | {record.get('status')} | {fmt(record.get('primary_value'))} | "
            f"{(f'{delta * 100:+.2f}%' if isinstance(delta, (int, float)) else '-')} | {record.get('kernel_count') or '-'} | "
            f"{', '.join(record.get('techniques') or []) or '-'} | {record.get('target') or '-'} | "
            f"{(record.get('description') or '').replace('|', '/')[:120]} |\n")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def add_note(path: Path, text: str, tags: list[str] | None = None, experiment: int | None = None) -> None:
    ensure(path)
    content = path.read_text(encoding="utf-8")
    marker = "## Experiment log"
    note = f"- {now_iso()[:16]}" + (f" (exp #{experiment})" if experiment is not None else "") + \
        (f" [{', '.join(tags)}]" if tags else "") + f": {text.strip()}\n"
    if marker in content:
        head, tail = content.split(marker, 1)
        content = head.rstrip("\n") + "\n" + note + "\n" + marker + tail
    else:
        content += "\n" + note
    path.write_text(content, encoding="utf-8")


def read_insights(path: Path, limit: int = 40) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    section = text.split("## Insights", 1)[-1].split("## Experiment log", 1)[0]
    lines = [line[2:].strip() for line in section.splitlines() if line.startswith("- ")]
    return lines[-limit:]
