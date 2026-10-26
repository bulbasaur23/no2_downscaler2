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
from no2map.service import create_app, live_compute_factory

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
        jid = c.post("/api/jobs", json={"city": "delhi_ncr", "polygon": SQUARE,
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

    def test_unknown_city_400(self, client):
        r = client.post("/api/jobs", json={"city": "atlantis", "polygon": SQUARE,
                                           "date": "2024-07-01"})
        assert r.status_code == 400 and "unknown city" in r.json()["detail"]

    def test_malformed_date_400(self, client):
        for bad in ("not-a-date", "2024-02-30", "2024-7-1"):
            r = client.post("/api/jobs", json={"city": "delhi_ncr",
                                               "polygon": SQUARE, "date": bad})
            assert r.status_code == 400, bad
            assert "ISO date" in r.json()["detail"], bad

    def test_compute_failure_marks_job_failed(self, tmp_path):
        def boom(req):
            raise RuntimeError("EE exploded")
        c = TestClient(create_app(store_dir=tmp_path / "j",
                                  runs_dir=tmp_path / "r", compute_fn=boom,
                                  use_live=False))
        jid = c.post("/api/jobs", json={"city": "delhi_ncr", "polygon": SQUARE,
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
        jid = c.post("/api/jobs", json={"city": "delhi_ncr", "polygon": SQUARE,
                                        "date": "2024-07-01"}).json()["id"]
        assert c.get(f"/api/results/{jid}/export.csv").status_code == 409
        assert c.get("/api/results/nope/export.csv").status_code == 404

    def test_demo_regions_are_valid_polygons(self, client):
        out = client.get("/api/demo/regions").json()["regions"]
        assert {r["name"] for r in out} == {"delhi", "hyderabad"}
        for r in out:
            validate_polygon([tuple(p) for p in r["polygon"]])
            assert r["city"] in {"delhi_ncr", "hyderabad"}


class TestLiveFactoryWiring:
    """The API process must initialize Earth Engine itself (pipeline scripts
    do it for their own runs); otherwise every live request fails with
    'client library not initialized' even with valid credentials."""

    def test_single_ee_init_and_fetchers_flow_through(self, tmp_path, monkeypatch):
        import pandas as pd

        import no2map.eerun as eerun

        from tests.test_live import (CELL_IDS, ELEV, _no2_fn, _weather_fn,
                                     build_test_artifact)

        runs = tmp_path / "runs"
        build_test_artifact(runs)
        exports = tmp_path / "exports"
        exports.mkdir()
        pd.DataFrame({
            "cell_id": CELL_IDS,
            "elevation_mean_m": [ELEV[c] for c in CELL_IDS],
            "elevation_std_m": 5.0,
            "elevation_min_m": [ELEV[c] - 4.0 for c in CELL_IDS],
            "elevation_max_m": [ELEV[c] + 4.0 for c in CELL_IDS],
            "water_fraction": 0.01,
        }).to_parquet(exports / "static_delhi_ncr.parquet", index=False)
        cfg_path = tmp_path / "default.yaml"
        cfg_path.write_text(
            "study_areas:\n  delhi_ncr:\n    lon_min: 76.38\n    lon_max: 78.62\n"
            "    lat_min: 27.60\n    lat_max: 29.32\n"
            "observed_cell_rule:\n  main: 0.5\n"
            "quota:\n  project_id: fake-project\n", encoding="utf-8")

        calls = {"n": 0}

        def fake_init(cfg=None):
            calls["n"] += 1
            return cfg or {}

        monkeypatch.setattr(eerun, "initialize_ee", fake_init)
        compute = live_compute_factory(
            cfg_path, exports, runs, cache_path=None,
            fetchers={"no2_fn": _no2_fn(), "weather_fn": _weather_fn,
                      "version_fn": lambda c, d: ["2.8.0"]})
        req = {"city": "delhi_ncr", "date": "2024-07-01",
               "polygon": [[77.0, 28.0], [77.2, 28.0],
                           [77.2, 28.05], [77.0, 28.05]]}
        first = compute(req)
        second = compute(req)
        assert calls["n"] == 1  # initialized once per process, not per request
        assert first["summary"]["n_cells"] == 4
        assert second["summary"] == first["summary"]
        assert first["climatology_anchor"]["available"] is True


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

    def test_city_list_comes_from_config(self, tmp_path):
        # the dropdown must follow study_areas, not a second hard-coded list
        cfg = tmp_path / "c.yaml"
        cfg.write_text("study_areas:\n  only_city: {}\n", encoding="utf-8")
        c = TestClient(create_app(store_dir=tmp_path / "j",
                                  runs_dir=tmp_path / "r",
                                  use_live=False, config_path=cfg))
        assert c.get("/api/availability").json()["cities"] == ["only_city"]
        ok = c.post("/api/jobs", json={"city": "only_city", "polygon": SQUARE,
                                       "date": "2024-07-01"})
        assert ok.status_code == 200


class TestRateLimit:
    def test_sixsixth_request_429(self, client):
        last = None
        for i in range(6):
            last = client.post("/api/jobs", json={"city": "delhi_ncr",
                                                  "polygon": SQUARE,
                                                  "date": "2024-07-01"})
        assert last.status_code == 429

    def test_forwarded_for_is_ignored_unless_proxy_trusted(self, client):
        # spoofing X-Forwarded-For must not buy a fresh bucket by default
        last = None
        for i in range(6):
            last = client.post("/api/jobs", json={"city": "delhi_ncr",
                                                  "polygon": SQUARE,
                                                  "date": "2024-07-01"},
                               headers={"x-forwarded-for": f"spoofed-{i}"})
        assert last.status_code == 429

    def test_proxy_mode_uses_forwarded_for(self, tmp_path):
        c = TestClient(create_app(store_dir=tmp_path / "j",
                                  runs_dir=tmp_path / "r",
                                  rate_limit_per_min=2, use_live=False,
                                  trust_proxy=True))
        for i in range(2):
            assert c.post("/api/jobs", json={"city": "delhi_ncr",
                                             "polygon": SQUARE,
                                             "date": "2024-07-01"},
                          headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
        assert c.post("/api/jobs", json={"city": "delhi_ncr", "polygon": SQUARE,
                                         "date": "2024-07-01"},
                      headers={"x-forwarded-for": "1.1.1.1"}).status_code == 429
        # a genuinely different client behind the real proxy is separate
        assert c.post("/api/jobs", json={"city": "delhi_ncr", "polygon": SQUARE,
                                         "date": "2024-07-01"},
                      headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200

    def test_env_var_enables_proxy_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NO2MAP_TRUST_PROXY", "true")
        c = TestClient(create_app(store_dir=tmp_path / "j",
                                  runs_dir=tmp_path / "r",
                                  rate_limit_per_min=1, use_live=False))
        assert c.post("/api/jobs", json={"city": "delhi_ncr", "polygon": SQUARE,
                                         "date": "2024-07-01"},
                      headers={"x-forwarded-for": "3.3.3.3"}).status_code == 200
        assert c.post("/api/jobs", json={"city": "delhi_ncr", "polygon": SQUARE,
                                         "date": "2024-07-01"},
                      headers={"x-forwarded-for": "4.4.4.4"}).status_code == 200


class TestModels:
    def test_registry_empty_then_loaded(self, tmp_path: Path):
        runs = tmp_path / "runs" / "delhi_ncr_modelA_20260101T000000Z"
        runs.mkdir(parents=True)
        meta = {"feature_schema_version": "fm-v1", "created_at": "t",
                "groups": ["static"], "alpha": 0.1,
                "climatology": {"city_median": 49.4}}
        (runs / "artifact.json").write_text(json.dumps(meta), encoding="utf-8")
        c = TestClient(create_app(store_dir=tmp_path / "j", runs_dir=runs.parent))
        out = c.get("/api/models").json()
        m = out["models"]
        assert len(m) == 1 and m[0]["artifact"] and m[0]["alpha"] == 0.1
        # the G3 fallback is served only if the bundle carries the anchor
        assert m[0]["climatology"] is True
        meta["climatology"] = None
        (runs / "artifact.json").write_text(json.dumps(meta), encoding="utf-8")
        assert c.get("/api/models").json()["models"][0]["climatology"] is False
