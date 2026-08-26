"""SQLite (WAL) store for experiments, live events, agent activity and worker leases.

Every writer (eval, driver, workers, dashboard controls) appends events; the dashboard tails
them over SSE. Records are stored as JSON blobs so schema evolution stays trivial.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .util import now_iso

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS experiments (
    number INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    parent INTEGER,
    commit_hash TEXT,
    description TEXT,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agents (
    name TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    detail TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS leases (
    target_id TEXT PRIMARY KEY,
    worker TEXT NOT NULL,
    state TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---- experiments -------------------------------------------------------------------
    def save_experiment(self, record: dict[str, Any]) -> None:
        now = now_iso()
        with self._lock:
            self._db.execute(
                """INSERT INTO experiments(number, name, status, parent, commit_hash, description, record_json,
                   created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(number) DO UPDATE SET name=excluded.name, status=excluded.status,
                   parent=excluded.parent, commit_hash=excluded.commit_hash, description=excluded.description,
                   record_json=excluded.record_json, updated_at=excluded.updated_at""",
                (int(record["number"]), record.get("name", ""), record.get("status", "planned"),
                 record.get("parent"), record.get("commit"), record.get("description", ""),
                 json.dumps(record, default=str), record.get("created_at", now), now),
            )
            self._db.commit()

    def get_experiment(self, number: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT record_json FROM experiments WHERE number=?", (number,)).fetchone()
        return json.loads(row["record_json"]) if row else None

    def list_experiments(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT record_json FROM experiments ORDER BY number ASC"
        with self._lock:
            rows = self._db.execute(query).fetchall()
        records = [json.loads(row["record_json"]) for row in rows]
        return records[-limit:] if limit else records

    def next_experiment_number(self) -> int:
        with self._lock:
            row = self._db.execute("SELECT MAX(number) AS m FROM experiments").fetchone()
        return (row["m"] + 1) if row and row["m"] is not None else 0

    # ---- events ------------------------------------------------------------------------
    def event(self, kind: str, **payload: Any) -> int:
        with self._lock:
            cur = self._db.execute("INSERT INTO events(ts, kind, payload_json) VALUES (?,?,?)",
                                   (now_iso(), kind, json.dumps(payload, default=str)))
            self._db.commit()
            return int(cur.lastrowid)

    def events_after(self, after_id: int, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT id, ts, kind, payload_json FROM events WHERE id > ? ORDER BY id ASC LIMIT ?",
                                    (after_id, limit)).fetchall()
        return [{"id": r["id"], "ts": r["ts"], "kind": r["kind"], "payload": json.loads(r["payload_json"])} for r in rows]

    def last_event_id(self) -> int:
        with self._lock:
            row = self._db.execute("SELECT MAX(id) AS m FROM events").fetchone()
        return int(row["m"] or 0)

    # ---- agents ------------------------------------------------------------------------
    def set_agent(self, name: str, state: str, detail: str = "") -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO agents(name, state, detail, updated_at) VALUES (?,?,?,?)
                   ON CONFLICT(name) DO UPDATE SET state=excluded.state, detail=excluded.detail, updated_at=excluded.updated_at""",
                (name, state, detail, now_iso()))
            self._db.commit()
        self.event("agent", name=name, state=state, detail=detail)

    def agents(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT name, state, detail, updated_at FROM agents ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    # ---- leases (parallel workers) -----------------------------------------------------
    def acquire_lease(self, target_id: str, worker: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT worker, state FROM leases WHERE target_id=?", (target_id,)).fetchone()
            if row and row["state"] == "active" and row["worker"] != worker:
                return False
            self._db.execute(
                """INSERT INTO leases(target_id, worker, state, updated_at) VALUES (?,?,'active',?)
                   ON CONFLICT(target_id) DO UPDATE SET worker=excluded.worker, state='active', updated_at=excluded.updated_at""",
                (target_id, worker, now_iso()))
            self._db.commit()
        self.event("lease", target=target_id, worker=worker, state="active")
        return True

    def release_lease(self, target_id: str, worker: str, state: str = "released") -> None:
        with self._lock:
            self._db.execute("UPDATE leases SET state=?, updated_at=? WHERE target_id=? AND worker=?",
                             (state, now_iso(), target_id, worker))
            self._db.commit()
        self.event("lease", target=target_id, worker=worker, state=state)

    def leases(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT target_id, worker, state, updated_at FROM leases").fetchall()
        return [dict(r) for r in rows]

    # ---- kv ----------------------------------------------------------------------------
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO kv(key, value_json, updated_at) VALUES (?,?,?)
                   ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at""",
                (key, json.dumps(value, default=str), now_iso()))
            self._db.commit()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._db.execute("SELECT value_json FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value_json"]) if row else default
