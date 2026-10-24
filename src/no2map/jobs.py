"""File-based job store (Phase 4, plan: POST /api/jobs -> file store).

Concept: polygon jobs are long-ish (EE composites take seconds to minutes),
so the API hands back an id immediately and work happens asynchronously.
A tiny file-per-job store (JSON under data/jobs/<id>.json) gives crash-safe
introspection, zero infrastructure, and survives server restarts - job
state lives on disk, not in process memory. Written atomically per update.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

_TERMINAL = {STATUS_DONE, STATUS_FAILED}


class JobStore:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def create(self, request: dict) -> dict:
        job = {
            "id": uuid.uuid4().hex[:12],
            "status": STATUS_QUEUED,
            "request": request,
            "result": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "updated_at": None,
        }
        self._write(job)
        return job

    def _write(self, job: dict) -> None:
        # atomic within the same directory (tempfile + rename), per update
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp).replace(self._path(job["id"]))

    def get(self, job_id: str) -> dict | None:
        p = self._path(job_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def update(self, job_id: str, *, status: str | None = None,
               result: dict | None = None, error: str | None = None) -> dict | None:
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return None
            if status is not None:
                if job["status"] in _TERMINAL and status not in _TERMINAL:
                    return job  # terminal states are stable
                job["status"] = status
            if result is not None:
                job["result"] = result
            if error is not None:
                job["error"] = error
            job["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._write(job)
            return job

    def list(self, limit: int = 50) -> list[dict]:
        jobs = [self.get(p.stem) for p in self.dir.glob("*.json")]
        jobs = [j for j in jobs if j]
        return sorted(jobs, key=lambda j: j["created_at"], reverse=True)[:limit]

    def counts(self) -> dict:
        out: dict[str, int] = {}
        for j in self.list(limit=10_000):
            out[j["status"]] = out.get(j["status"], 0) + 1
        return out
