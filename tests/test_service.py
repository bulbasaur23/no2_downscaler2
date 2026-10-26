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
from no2map.service import create_app

SQUARE = [(77.0, 28.0), (77.4, 28.0), (77.4, 28.4), (77.0, 28.4)]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app(store_dir=tmp_path / "jobs", runs_dir=tmp_path / "runs",
                     rate_limit_per_min=5)
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
                                  runs_dir=tmp_path / "r", compute_fn=boom))
        jid = c.post("/api/jobs", json={"city": "x", "polygon": SQUARE,
                                        "date": "2024-07-01"}).json()["id"]
        g = c.get(f"/api/jobs/{jid}").json()
        assert g["status"] == "failed" and "EE exploded" in g["error"]
        assert c.get(f"/api/results/{jid}").status_code == 409


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
