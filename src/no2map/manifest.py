"""Export manifest (Phase 1): records which city-months are done.

Batching design follows plan Phase 0/1 (Gate G0 fallback 2 and risk 3):
 - training data exports one city-month at a time to Parquet,
 - a manifest JSON records what is DONE, so an interrupted run resumes
   without re-spending Earth Engine compute (quota decision D5),
 - a month that fails twice is marked skipped and recorded - it never retried
   silently forever (plan Gate G1 fallback: 'skip months that fail twice and
   record them').

The manifest is the provenance backbone: every Parquet file references the
manifest entry that produced it, and every entry carries the feature/schema
and source versions so a month can be invalidated when versions change.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

MAX_ATTEMPTS = 2  # plan G1 fallback: fail twice -> skip and record

STATUS_DONE = "done"
STATUS_FAILED = "failed"      # failed but attempts remain
STATUS_SKIPPED = "skipped"    # exhausted attempts, recorded (never silent)
STATUS_PENDING = "pending"


@dataclass
class MonthEntry:
    city: str
    month: str  # "YYYY-MM"
    status: str = STATUS_PENDING
    attempts: int = 0
    parquet_path: str | None = None
    n_rows: int | None = None
    feature_schema_version: str | None = None
    processor_versions: list | None = None  # NO2 versions seen in this month (D3)
    last_error: str | None = None


@dataclass
class ExportManifest:
    path: Path
    entries: dict[str, MonthEntry] = field(default_factory=dict)

    # ---- keying ----------------------------------------------------------
    @staticmethod
    def key(city: str, month: str) -> str:
        return f"{city}/{month}"

    # ---- persistence (atomic: crash mid-write must not corrupt) ----------
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "entries": {k: asdict(v) for k, v in self.entries.items()}}
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        Path(tmp).replace(self.path)  # atomic on same volume

    @classmethod
    def load(cls, path: Path) -> "ExportManifest":
        m = cls(path=path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            m.entries = {k: MonthEntry(**v) for k, v in data.get("entries", {}).items()}
        return m

    # ---- month range -----------------------------------------------------
    @staticmethod
    def months_between(start: str, end: str) -> list[str]:
        y0, m0 = map(int, start.split("-"))
        y1, m1 = map(int, end.split("-"))
        out = []
        y, m = y0, m0
        while (y, m) <= (y1, m1):
            out.append(f"{y:04d}-{m:02d}")
            m += 1
            if m == 13:
                y, m = y + 1, 1
        return out

    # ---- lifecycle -------------------------------------------------------
    def ensure_pending(self, city: str, months: list[str]) -> list[str]:
        """Insert pending entries for months not yet tracked; return new ones."""
        new = []
        for month in months:
            k = self.key(city, month)
            if k not in self.entries:
                self.entries[k] = MonthEntry(city=city, month=month)
                new.append(month)
        return new

    def record_success(self, city: str, month: str, parquet_path: str, n_rows: int,
                       feature_schema_version: str, processor_versions: list) -> None:
        e = self.entries[self.key(city, month)]
        e.status = STATUS_DONE
        e.parquet_path = parquet_path
        e.n_rows = n_rows
        e.feature_schema_version = feature_schema_version
        e.processor_versions = processor_versions
        e.last_error = None

    def record_failure(self, city: str, month: str, error: str) -> str:
        """Count the attempt; after MAX_ATTEMPTS failures mark SKIPPED.
        Returns the new status so the caller can decide to stop the batch."""
        e = self.entries[self.key(city, month)]
        e.attempts += 1
        e.last_error = error[:2000]
        if e.attempts >= MAX_ATTEMPTS and e.status != STATUS_DONE:
            e.status = STATUS_SKIPPED
        else:
            e.status = STATUS_FAILED
        return e.status

    # ---- queries ---------------------------------------------------------
    def is_done(self, city: str, month: str) -> bool:
        e = self.entries.get(self.key(city, month))
        return bool(e and e.status == STATUS_DONE)

    def months_todo(self, city: str, months: list[str]) -> list[str]:
        """Months still needing work = pending + failed (retryable).
        Skipped months are DONE-in-spirit: they stay recorded, not retried."""
        out = []
        for m in months:
            e = self.entries.get(self.key(city, m))
            if e is None or e.status in (STATUS_PENDING, STATUS_FAILED):
                out.append(m)
        return out

    def schema_mismatches(self, city: str, feature_schema_version: str) -> list[str]:
        """Months done under a different schema version - re-export targets
        when the feature schema changes (parity guarantee, plan risk 10)."""
        return [
            e.month for k, e in self.entries.items()
            if e.city == city and e.status == STATUS_DONE and e.feature_schema_version != feature_schema_version
        ]
