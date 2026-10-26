"""Per-cell result cache, keyed exactly as the plan specifies.

Plan (Phase 4): "Cache keyed on cell ID, date, feature version, processor
version and model version." All five enter the key, so a retrained model or a
processor upgrade can never serve a stale cell: the key changes and the row is
recomputed. Storage is one SQLite file (stdlib, no service to run), which also
gives atomic writes and safe concurrent use from FastAPI's background threads.

Only finished per-cell rows are cached - never inputs - so a cache hit is
byte-identical to what the live path would have returned (same artifact, same
feature schema, same processor version).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cells (
    key TEXT PRIMARY KEY,
    cell_id TEXT NOT NULL,
    date TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    processor_versions TEXT NOT NULL,
    model_version TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


def cell_key(cell_id: str, date: str, feature_version: str,
             processor_versions: str | list[str], model_version: str) -> str:
    """Stable key over all five plan components (order fixed)."""
    if isinstance(processor_versions, (list, tuple)):
        processor_versions = ",".join(sorted(str(v) for v in processor_versions))
    raw = "\x1f".join([cell_id, date, feature_version,
                       str(processor_versions), model_version])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CellCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, dict]:
        if not keys:
            return {}
        marks = ",".join("?" for _ in keys)
        with self._lock:
            cur = self._conn.execute(
                f"SELECT key, payload FROM cells WHERE key IN ({marks})", keys)
            return {k: json.loads(p) for k, p in cur.fetchall()}

    def put_many(self, rows: list[tuple[str, str, str, str, str, str, dict]]) -> int:
        """rows: (key, cell_id, date, feature_version, processor_versions,
        model_version, payload)."""
        if not rows:
            return 0
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO cells (key, cell_id, date, feature_version,"
                " processor_versions, model_version, payload, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                [(k, cid, d, fv, pv, mv, json.dumps(p), now)
                 for k, cid, d, fv, pv, mv, p in rows])
            self._conn.commit()
        return len(rows)

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM cells").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
