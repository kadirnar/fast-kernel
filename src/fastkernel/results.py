"""results.tsv: the human-readable experiment ledger (autoresearch convention, tab separated)."""
from __future__ import annotations

from pathlib import Path

COLUMNS = ["exp", "commit", "status", "metric", "value", "speedup", "peak_vram_gb", "gates", "description"]


def ensure_header(path: Path) -> None:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        path.write_text("\t".join(COLUMNS) + "\n", encoding="utf-8")


def append_row(path: Path, *, exp: int, commit: str, status: str, metric: str, value: float | None,
               speedup: float | None, peak_vram_gb: float | None, gates: str, description: str) -> None:
    ensure_header(path)
    row = [
        str(exp), commit or "-", status, metric,
        f"{value:.6f}" if value is not None else "0.000000",
        f"{speedup:.3f}" if speedup is not None else "0.000",
        f"{peak_vram_gb:.2f}" if peak_vram_gb is not None else "0.00",
        gates, description.replace("\t", " ").replace("\n", " ").strip(),
    ]
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\t".join(row) + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        rows.append({header[i]: parts[i] if i < len(parts) else "" for i in range(len(header))})
    return rows
