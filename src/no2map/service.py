"""The Phase-4 API skeleton: polygon jobs over a file-based store.

Endpoints (per plan):
  POST /api/jobs      {city, polygon, date} -> {"id", "status"} (202-style quick ack)
  GET  /api/jobs/{id} -> job incl. status + result when done
  GET  /api/results/{id} -> just the result payload (404 until done)
  GET  /api/models    -> trained artifacts in data/runs (provenance data)

The compute function is INJECTED (compute_fn: request_dict -> result dict).
By default `create_app` wires the REAL path (`live_compute_factory`: EE fetch ->
shared features -> schema-pinned artifact), and the geometric-only `stub_compute`
remains as the explicit fallback when no artifact is available. Tests inject
their own compute_fn (or pass use_live=False) so the HTTP/job/validation/
rate-limit scaffolding is exercised without Earth Engine.

Rate limiting: sliding-window counter per client IP (plan: per-IP rate
limiting) - limits are constructor injected so tests can shrink them.
"""
from __future__ import annotations

import csv
import io
import json
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import (BackgroundTasks, FastAPI, HTTPException, Request,
                     Response)
from fastapi.responses import HTMLResponse

from no2map.jobs import JobStore
from no2map.live import DATE_CAP_LAG_DAYS, GRID_CELL_DEG, latest_available_date
from no2map.polygon import (
    MAX_VERTICES,
    cells_for_polygon,
    polygon_bbox_buffered,
    resolution_warning,
    validate_polygon,
)

ROOT = Path(__file__).resolve().parents[2]

# D7 demo mode: fixed polygons so the app can be exercised without drawing.
DEMO_REGIONS = {
    "delhi": {"city": "delhi_ncr",
              "polygon": [[76.95, 28.40], [77.45, 28.40],
                          [77.45, 28.80], [76.95, 28.80]]},
    "hyderabad": {"city": "hyderabad",
                  "polygon": [[78.25, 17.30], [78.60, 17.30],
                              [78.60, 17.60], [78.25, 17.60]]},
}


def stub_compute(request: dict) -> dict:
    """Geometric-only preview, used when no trained artifact is available.

    The real path is `live_compute`; this fallback keeps the HTTP/job flow
    usable (and honest - it says exactly what it did) before G3 artifacts or
    in a fresh checkout where data/runs is empty."""
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
        "note": "geometric preview only: no model artifact was loaded, so no "
                "values are reconstructed",
    }


def live_compute_factory(
    config_path: str | Path = ROOT / "configs" / "default.yaml",
    exports_dir: str | Path = ROOT / "data" / "exports",
    runs_dir: str | Path = ROOT / "data" / "runs",
    cache_path: str | Path | None = ROOT / "data" / "cache" / "cells.sqlite",
    fetchers: dict | None = None,
):
    """Build a request->result function backed by the EE live path.

    Predictors are created lazily and cached per city: each loads the newest
    complete artifact and the statics cache once, so a request only pays for
    the EE fetch. Configuration/artifact problems surface as job failures with
    the real message instead of a silently empty map. One shared CellCache
    implements the plan's five-part key (pass cache_path=None to disable).

    Earth Engine is initialized here (once, lazily) because the API process is
    not started by a pipeline script: without this every request would fail
    with "client library not initialized" even with valid credentials. The
    same failure would otherwise look like a data problem.

    `fetchers` forwards no2_fn/weather_fn/version_fn to LivePredictor - the
    seam tests use to exercise this wiring without network access.
    """
    import yaml

    from no2map.cache import CellCache
    from no2map.live import (LivePredictor, load_g3_summary,
                             load_latest_artifact, load_statics)

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cache = CellCache(cache_path) if cache_path else None
    cache_by_city: dict[str, LivePredictor] = {}
    lock = threading.Lock()
    ee_ready = False

    def _predictor(city: str) -> LivePredictor:
        nonlocal ee_ready
        with lock:
            if not ee_ready:
                from no2map.eerun import initialize_ee
                initialize_ee(cfg)  # raises here -> job fails with the cause
                ee_ready = True
            if city not in cache_by_city:
                art, run_name = load_latest_artifact(runs_dir, city)
                cache_by_city[city] = LivePredictor(
                    city=city,
                    city_cfg=cfg["study_areas"][city],
                    artifact=art,
                    statics=load_statics(exports_dir, city),
                    run_name=run_name,
                    main_threshold=float(
                        cfg["observed_cell_rule"]["main"]),
                    cache=cache,
                    g3=load_g3_summary(exports_dir, city),
                    **(fetchers or {}),
                )
            return cache_by_city[city]

    def compute(request: dict) -> dict:
        city = request["city"]
        if city not in cfg["study_areas"]:
            raise ValueError(f"unknown city {city!r}")
        return _predictor(city).compute(
            [tuple(p) for p in request["polygon"]], request["date"])

    return compute


def create_app(
    store_dir: str | Path = ROOT / "data" / "jobs",
    runs_dir: str | Path = ROOT / "data" / "runs",
    compute_fn=None,
    rate_limit_per_min: int = 30,
    exports_dir: str | Path = ROOT / "data" / "exports",
    config_path: str | Path = ROOT / "configs" / "default.yaml",
    use_live: bool = True,
) -> FastAPI:
    """API factory. `use_live=True` (default) serves real predictions; tests
    inject `compute_fn` (or pass use_live=False) to exercise the flow without
    Earth Engine or a trained artifact."""
    if compute_fn is None:
        compute_fn = (live_compute_factory(config_path, exports_dir, runs_dir)
                      if use_live else stub_compute)
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

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = Path(__file__).with_name("web") / "index.html"
        if not page.exists():
            raise HTTPException(404, "frontend not installed")
        return HTMLResponse(page.read_text(encoding="utf-8"))

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
        cap = latest_available_date()
        if str(date) > cap:
            raise HTTPException(
                400, f"date {date} is beyond the availability cap {cap} "
                     f"(D4: sources lag {DATE_CAP_LAG_DAYS} days)")
        job = store.create({"city": city, "polygon": polygon, "date": date})
        bg.add_task(_run_job, job["id"], store, compute_fn)
        return {"id": job["id"], "status": job["status"]}

    @app.get("/api/availability")
    def availability():
        """D4 'data as of' line: the newest date the picker may offer."""
        return {"latest_date": latest_available_date(),
                "date_cap_lag_days": DATE_CAP_LAG_DAYS,
                "grid_cell_deg": GRID_CELL_DEG,
                "cities": ["delhi_ncr", "bengaluru", "kolkata", "hyderabad"]}

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

    @app.get("/api/results/{job_id}/export.csv")
    def export_csv(job_id: str):
        """CSV export with the status column (plan: GeoTIFF may be cut first,
        CSV stays). Always reports the cell state, so observed and
        reconstructed rows are never conflated in downstream use."""
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if job["status"] != "done":
            raise HTTPException(409, f"job not done (status={job['status']})")
        result = job["result"]
        cols = ["cell_id", "lon", "lat", "weight", "state", "reason_code",
                "source", "value", "value_observed", "pred", "climatology",
                "margin", "interval_lo", "interval_hi", "valid_fraction"]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(result.get("cells") or [])
        fname = f"no2_{result.get('city', 'city')}_{result.get('date', 'date')}.csv"
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{fname}"'})

    @app.get("/api/demo/regions")
    def demo_regions():
        return {"regions": [{"name": k, **v} for k, v in DEMO_REGIONS.items()]}

    @app.get("/api/models")
    def list_models():
        runs = sorted(Path(runs_dir).glob("*_modelA_*")) if Path(runs_dir).exists() else []
        out = []
        for d in runs:
            art = d / "artifact.json"
            if not art.exists():
                out.append({"run": d.name, "artifact": False})
                continue
            meta = json.loads(art.read_text(encoding="utf-8"))
            out.append({"run": d.name, "artifact": True,
                        "feature_schema_version": meta.get("feature_schema_version"),
                        "created_at": meta.get("created_at"),
                        "groups": meta.get("groups"),
                        "alpha": meta.get("alpha"),
                        # G3 honest-report: is the climatology anchor served?
                        "climatology": meta.get("climatology") is not None})
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
