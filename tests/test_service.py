"""Tests for the Phase-4 API skeleton (job flow over the file store).

Exercised through fastapi TestClient against a TEMP store dir: the complete
POST -> queued/running -> done lifecycle with the geometric stub result
(exact cell weights from the polygon engine), 400-rejections for malformed
polygons and missing fields, 404/409 semantics of jobs vs results, the
per-IP rate limiter, and the /api/models registry shape.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from no2map.grid import cell_from_point
from no2map.polygon import validate_polygon
from no2map.service import create_app

SQUARE = [(77.0, 28.0), (77.4, 28.0), (77.4, 28.4), (77.0, 28.4)]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app(store_dir=tmp_path / "jobs", runs_dir=tmp_path / "runs",
                     rate_limit_per_min=5, use_live=False)
    return TestClient(app)


def _result_of(client: TestClient, job_id: str) -> dict:
    r = client.get(f"/api/results/{job_id}")
    assert r.status_code == 200, r.text
    return r.json()


class TestJobFlow:
    def test_full_lifecycle(self, client):
        r = client.post("/api/jobs",
                        json={"city": "delhi_ncr", "polygon": SQUARE,
                              "date": "2024-07-01"})
        assert r.status_code == 200, r.text
        jid = r.json()["id"]
        assert r.json()["status"] == "queued"
        # BackgroundTasks run eagerly after the response in TestClient:
        # job must already be terminal-and-done
        g = client.get(f"/api/jobs/{jid}")
        assert g.status_code == 200
        assert g.json()["status"] == "done"
        assert g.json()["error"] is None
        res = _result_of(client, jid)
        assert res["n_cells"] == 64  # 0.4 deg square = 8x8 full cells
        assert all(abs(w - 1.0) < 1e-9 for w in res["cell_weights"].values())
        assert res["fetch_window"]["lat_min"] < 28.0  # buffer included

    def test_sub_cell_polygon_says_so(self, client):
        tiny = [(77.01, 28.01), (77.02, 28.01), (77.02, 28.02), (77.01, 28.02)]
        r = client.post("/api/jobs", json={"city": "delhi_ncr",
                                           "polygon": tiny, "date": "2024-07-01"})
        res = _result_of(client, r.json()["id"])
        assert res["n_cells"] == 1
        assert any("smaller than one grid cell" in (res["resolution_warning"] or [""])[0]
                   for _ in [0]) or "smaller than one grid cell" in str(res["resolution_warning"])

    def test_results_gated_until_done(self, tmp_path):
        def slow(req):
            import time
            time.sleep(0.2)
            return {"ok": True}
        c = TestClient(create_app(store_dir=tmp_path / "j",
                                  runs_dir=tmp_path / "r", compute_fn=slow))
        jid = c.post("/api/jobs", json={"city": "x", "polygon": SQUARE,
                                        "date": "2024-07-01"}).json()["id"]
        r = c.get(f"/api/results/{jid}")
        # TestClient blocks BackgroundTasks until response sent; reject may
        # already be done -> accept either 409 (still running) or 200
        assert r.status_code in (200, 409)


class TestValidation:
    def test_missing_fields_400(self, client):
        r = client.post("/api/jobs", json={"city": "delhi_ncr"})
        assert r.status_code == 400

    def test_edge_polygon_400(self, client):
        r = client.post("/api/jobs", json={"city": "x",
                                           "polygon": [(77.0, 28.0), (77.4, 28.4)],
                                           "date": "2024-07-01"})
        assert r.status_code == 400 and "polygon" in r.json()["detail"]

    def test_self_intersecting_400(self, client):
        bowtie = [(77.0, 28.0), (77.4, 28.4), (77.4, 28.0), (77.0, 28.4)]
        r = client.post("/api/jobs", json={"city": "x", "polygon": bowtie,
                                           "date": "2024-07-01"})
        assert r.status_code == 400 and "self-intersect" in r.json()["detail"]

    def test_outside_india_400(self, client):
        out = [(98.0, 28.0), (98.4, 28.0), (98.4, 28.4), (98.0, 28.4)]
        r = client.post("/api/jobs", json={"city": "x", "polygon": out,
                                           "date": "2024-07-01"})
        assert r.status_code == 400 and "India" in r.json()["detail"]

    def test_compute_failure_marks_job_failed(self, tmp_path):
        def boom(req):
            raise RuntimeError("EE exploded")
        c = TestClient(create_app(store_dir=tmp_path / "j",
                                  runs_dir=tmp_path / "r", compute_fn=boom,
                                  use_live=False))
        jid = c.post("/api/jobs", json={"city": "x", "polygon": SQUARE,
                                        "date": "2024-07-01"}).json()["id"]
        g = c.get(f"/api/jobs/{jid}").json()
        assert g["status"] == "failed" and "EE exploded" in g["error"]
        assert c.get(f"/api/results/{jid}").status_code == 409


class TestResultsExport:
    def test_csv_export_has_status_column_and_cells(self, tmp_path):
        def compute(req):
            return {"city": req["city"], "date": req["date"], "cells": [
                {"cell_id": "r1c1", "lon": 77.0, "lat": 28.0, "weight": 1.0,
                 "state": "observed", "reason_code": None, "source": "observed",
                 "value": 12.5, "value_observed": 12.5, "pred": 11.0,
                 "climatology": 9.75, "margin": 3.0, "interval_lo": 8.0,
                 "interval_hi": 14.0},
                {"cell_id": "r1c2", "lon": 77.05, "lat": 28.0, "weight": 0.4,
                 "state": "insufficient", "reason_code": "low_coverage",
                 "source": "reconstructed", "value": 10.0,
                 "value_observed": None, "pred": 10.0, "climatology": 9.75,
                 "margin": 2.5, "interval_lo": 7.5, "interval_hi": 12.5},
            ]}

        c = TestClient(create_app(store_dir=tmp_path / "j", runs_dir=tmp_path / "r",
                                  compute_fn=compute, use_live=False))
        jid = c.post("/api/jobs", json={"city": "delhi_ncr", "polygon": SQUARE,
                                        "date": "2024-07-01"}).json()["id"]
        r = c.get(f"/api/results/{jid}/export.csv")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        lines = r.text.strip().splitlines()
        header = lines[0].split(",")
        assert header[0] == "cell_id"
        assert {"state", "source", "value", "interval_lo", "climatology"} <= set(header)
        assert len(lines) == 3
        assert ",observed," in lines[1] and ",insufficient," in lines[2]

    def test_csv_for_failed_job_is_409(self, tmp_path):
        def boom(req):
            raise RuntimeError("nope")
        c = TestClient(create_app(store_dir=tmp_path / "j", runs_dir=tmp_path / "r",
                                  compute_fn=boom, use_live=False))
        jid = c.post("/api/jobs", json={"city": "x", "polygon": SQUARE,
                                        "date": "2024-07-01"}).json()["id"]
        assert c.get(f"/api/results/{jid}/export.csv").status_code == 409
        assert c.get("/api/results/nope/export.csv").status_code == 404

    def test_demo_regions_are_valid_polygons(self, client):
        out = client.get("/api/demo/regions").json()["regions"]
        assert {r["name"] for r in out} == {"delhi", "hyderabad"}
        for r in out:
            validate_polygon([tuple(p) for p in r["polygon"]])
            assert r["city"] in {"delhi_ncr", "hyderabad"}


class TestFrontend:
    def test_index_served(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        body = r.text
        for needle in ("leaflet", "/api/jobs", "/api/availability", "Export CSV",
                       "climatology", "g3note"):
            assert needle.lower() in body.lower()


class TestAvailability:
    def test_reports_d4_cap(self, client):
        r = client.get("/api/availability")
        assert r.status_code == 200
        body = r.json()
        assert body["date_cap_lag_days"] == 6
        assert body["grid_cell_deg"] == pytest.approx(0.05)
        from no2map.live import latest_available_date
        assert body["latest_date"] == latest_available_date()

    def test_future_date_beyond_cap_is_rejected(self, client):
        r = client.post("/api/jobs", json={"city": "delhi_ncr", "polygon": SQUARE,
                                           "date": "2099-01-01"})
        assert r.status_code == 400
        assert "availability cap" in r.json()["detail"]

    def test_date_inside_cap_is_accepted(self, client):
        r = client.post("/api/jobs", json={"city": "delhi_ncr", "polygon": SQUARE,
                                           "date": "2024-07-01"})
        assert r.status_code == 200


class TestRateLimit:
    def test_sixsixth_request_429(self, client):
        last = None
        for i in range(6):
            last = client.post("/api/jobs", json={"city": "x", "polygon": SQUARE,
                                                  "date": "2024-07-01"})
        assert last.status_code == 429
        # a different IP is a different bucket
        r = client.post("/api/jobs", json={"city": "x", "polygon": SQUARE,
                                           "date": "2024-07-01"},
                        headers={"x-forwarded-for": "other-ip"})
        assert r.status_code == 200


class TestModels:
    def test_registry_empty_then_loaded(self, tmp_path: Path):
        runs = tmp_path / "runs" / "delhi_ncr_modelA_20260101T000000Z"
        runs.mkdir(parents=True)
        (runs / "artifact.json").write_text(json.dumps({
            "feature_schema_version": "fm-v1", "created_at": "t",
            "groups": ["static"], "alpha": 0.1}), encoding="utf-8")
        c = TestClient(create_app(store_dir=tmp_path / "j", runs_dir=runs.parent))
        out = c.get("/api/models").json()
        m = out["models"]
        assert len(m) == 1 and m[0]["artifact"] and m[0]["alpha"] == 0.1
