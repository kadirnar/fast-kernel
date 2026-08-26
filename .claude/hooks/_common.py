"""Shared helpers for fast-kernel hooks (stdlib only; the hooks must not import fastkernel)."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

PROTECTED = ("GOAL.md", "spec.py", "results.tsv", "experiments", ".fast-kernel", "harness")
LIBRARY_PROTECTED = ("src/fastkernel/harness", "src/fastkernel/models", "src/fastkernel/profiling", "src/fastkernel/config.py",
                     "src/fastkernel/campaign.py")


def read_input() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:  # noqa: BLE001
        return {}


def project_dir() -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()).resolve()


def find_campaigns(root: Path, max_depth: int = 3) -> list[Path]:
    found = []
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__")) and d not in ("node_modules", "site-packages")]
        if "GOAL.md" in filenames and "candidate" in dirnames:
            found.append(Path(dirpath))
            dirnames[:] = []
        elif depth >= max_depth:
            dirnames[:] = []
    return found


def campaign_of(path: Path) -> Path | None:
    for candidate in [path, *path.parents]:
        if (candidate / "GOAL.md").exists() and (candidate / "candidate").is_dir():
            return candidate
    return None


def experiment_count(campaign: Path) -> int:
    db = campaign / ".fast-kernel" / "state.db"
    if not db.exists():
        return 0
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        try:
            row = con.execute("SELECT COUNT(*), MAX(number) FROM experiments").fetchone()
            return int(row[0] or 0)
        finally:
            con.close()
    except sqlite3.Error:
        return 0


def last_experiment(campaign: Path) -> dict | None:
    db = campaign / ".fast-kernel" / "state.db"
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        try:
            row = con.execute("SELECT record_json FROM experiments ORDER BY number DESC LIMIT 1").fetchone()
            return json.loads(row[0]) if row else None
        finally:
            con.close()
    except sqlite3.Error:
        return None
