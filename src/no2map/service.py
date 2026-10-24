"""The Phase-4 API skeleton: polygon jobs over a file-based store.

Endpoints (per plan):
  POST /api/jobs      {city, polygon, date} -> {"id", "status"} (202-style quick ack)
  GET  /api/jobs/{id} -> job incl. status + result when done
  GET  /api/results/{id} -> just the result payload (404 until done)
  GET  /api/models    -> trained artifacts in data/runs (provenance data)

The compute function is INJECTED (compute_fn: request_dict -> result dict).
The real EE-feature fetch will be plugged here after G1; with the default
stub the job emits the polygon-compute plan: validated polygon stats, the
covered cells with their area-overlap fractions (exact, Sutherland-Hodgman),
and the buffered fetch window cells will ACT on. This keeps the HTTP/job/
validation/rate-limit scaffolding fully testable before GEE wiring.

Rate limiting: sliding-window counter per client IP (plan: per-IP rate
limiting) - limits are constructor injected so tests can shrink them.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from no2map.jobs import JobStore
from no2map.polygon import (
    MAX_VERTICES,
    cells_for_polygon,
    polygon_bbox_buffered,
    resolution_warning,
    validate_polygon,
)

ROOT = Path(__file__).resolve().parents[2]


def stub_compute(request: dict) -> dict:
    """Posts the geometric part of the Phase-4 flow; replaced by the real
    EE feature fetch + artifact prediction once data is live."""
    info = validate_polygon(request["polygon"])
    overlaps = cells_for_polygon(request["polygon"])
    window = polygon_bbox_buffered(request["polygon"])
    min_f = min(overlaps.values()) if overlaps else 0.0
    return {
        "polygon": {
            "area_m2": info["area_m2"],
            "n_vertices": info["n_vertices"],
        },
        "resolution_warning": (info["warnings"] +
                               [w for w in [resolution_warning(len(overlaps), min_f)]
                               if w]) or None,
        "n_cells": len(overlaps),
        "cell_weights": overlaps,
        "fetch_window": window,
        "date": request.get("date"),
        "note": "geometric preview: EE features + model prediction land after Gate G1/G3",
    }


def create_app(
    store_dir: str | Path = ROOT / "data" / "jobs",
    runs_dir: str | Path = ROOT / "data" / "runs",
    compute_fn=stub_compute,
    rate_limit_per_min: int = 30,
) -> FastAPI:
    app = FastAPI(title="NO2 mapping platform", version="0.1.0")
    store = JobStore(store_dir)

    # ---- per-IP sliding-window rate limiter ------------------------------
    hits: dict[str, deque[float]] = defaultdict(deque)
    hits_lock = threading.Lock()

    def _allow(ip: str) -> bool:
        now = time.monotonic()
        with hits_lock:
            q = hits[ip]
            while q and now - q[0] > 60.0:
                q.popleft()
            if len(q) >= rate_limit_per_min:
                return False
            q.append(now)
            return True

    def _client_ip(req: Request) -> str:
        fwd = req.headers.get("x-forwarded-for")
        return (fwd.split(",")[0].strip() if fwd else
                req.client.host if req.client else "anon")

    # ---- endpoints --------------------------------------------------------

    @app.post("/api/jobs")
    def post_job(req: Request, body: dict, bg: BackgroundTasks):
        if not _allow(_client_ip(req)):
            raise HTTPException(429, "rate limit exceeded; retry later")
        city = body.get("city")
        polygon = body.get("polygon")
        date = body.get("date")
        if not city or not polygon or not date:
            raise HTTPException(400, "request needs 'city', 'polygon', 'date'")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise HTTPException(400, "polygon must be a list of >=3 [lon,lat] pairs")
        if len({tuple(p) for p in polygon}) > MAX_VERTICES:
            raise HTTPException(400, f"polygon exceeds {MAX_VERTICES} vertices")
        try:
            validate_polygon([tuple(p) for p in polygon])
        except ValueError as e:
            raise HTTPException(400, f"invalid polygon: {e}") from e
        job = store.create({"city": city, "polygon": polygon, "date": date})
        bg.add_task(_run_job, job["id"], store, compute_fn)
        return {"id": job["id"], "status": job["status"]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return {"id": job["id"], "status": job["status"], "error": job["error"],
                "created_at": job["created_at"], "updated_at": job["updated_at"]}

    @app.get("/api/results/{job_id}")
    def get_result(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if job["status"] != "done":
            raise HTTPException(409, f"job not done (status={job['status']})")
        return job["result"]

    @app.get("/api/models")
    def list_models():
        runs = sorted(Path(runs_dir).glob("*_modelA_*")) if Path(runs_dir).exists() else []
        out = []
        for d in runs:
            art = d / "artifact.json"
            if not art.exists():
                out.append({"run": d.name, "artifact": False})
                continue
            meta = __import__("json").loads(art.read_text(encoding="utf-8"))
            out.append({"run": d.name, "artifact": True,
                        "feature_schema_version": meta.get("feature_schema_version"),
                        "created_at": meta.get("created_at"),
                        "groups": meta.get("groups"),
                        "alpha": meta.get("alpha")})
        return {"models": out}

    return app


def _run_job(job_id: str, store: JobStore, compute_fn) -> None:
    store.update(job_id, status="running")
    try:
        job = store.get(job_id)
        req = job["request"]
        result = compute_fn(req)
        store.update(job_id, status="done", result=result)
    except ValueError as e:  # polygon validation bled through - report, not crash
        store.update(job_id, status="failed", error=str(e))
    except Exception as e:  # noqa: BLE001 - job failure must be recorded, not raised
        store.update(job_id, status="failed",
                     error=f"{type(e).__name__}: {e}")
