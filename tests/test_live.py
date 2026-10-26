"""Tests for the live polygon compute path (EE fetch -> features -> artifact).

EE is faked through the injectable fetchers, so these tests pin the parts that
must be right regardless of the network: exact cell coverage weighting, D6
state propagation into the observed/reconstructed labels, weighted summary
arithmetic, honest handling of cells the fetch never returned, the D4 date cap,
and the schema pin on the loaded artifact.
"""
from __future__ import annotations

from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from no2map.artifact import save_model_artifact
from no2map.conformal import SplitConformal
from no2map.features import (FEATURE_GROUPS, TARGET, build_features,
                             feature_block)
from no2map.grid import cell_from_point
from no2map.live import (LivePredictor, latest_available_date,
                         load_latest_artifact, load_statics)
from no2map.model import XGBConfig, season_key

GROUPS = list(FEATURE_GROUPS)
POLY = [(77.0, 28.0), (77.2, 28.0), (77.2, 28.05), (77.0, 28.05)]
CELL_IDS = [cell_from_point(28.02, lon).id for lon in (77.02, 77.07, 77.12, 77.17)]
ELEV = {cid: 200.0 + 10.0 * i for i, cid in enumerate(CELL_IDS)}


def _training_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(3)
    for month in (6, 7):
        for day in range(1, 11):
            d = f"2024-{month:02d}-{day:02d}"
            for cid in CELL_IDS:
                rows.append(dict(
                    cell_id=cid, date=d, state="observed",
                    **{TARGET: 0.1 * ELEV[cid] + 2.0 * month + rng.normal(0, 1.0)},
                    elevation_mean_m=ELEV[cid], elevation_std_m=5.0,
                    elevation_min_m=ELEV[cid] - 4.0, elevation_max_m=ELEV[cid] + 4.0,
                    water_fraction=0.01,
                    wind_u10_inst=rng.normal(1, 0.5), wind_v10_inst=rng.normal(1, 0.5),
                    wind_u10_l24=rng.normal(1, 0.5), wind_v10_l24=rng.normal(1, 0.5),
                    wind_u10_l48=rng.normal(1, 0.5), wind_v10_l48=rng.normal(1, 0.5),
                    t2m_k_inst=300.0 + rng.normal(0, 0.5),
                    d2m_k_inst=292.0 + rng.normal(0, 0.5),
                    sp_pa_inst=100_000.0, blh_m_inst=900.0 + rng.normal(0, 50),
                    precip_l24_m=0.0, precip_l48_m=0.0,
                ))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("live")
    df = _training_frame()
    built = build_features(df)
    fe_cols = feature_block(built, GROUPS)
    model = XGBConfig(n_estimators=40).make(seed=0)
    model.fit(built[fe_cols], built[TARGET])
    resid = built[TARGET] - model.predict(built[fe_cols])
    conf = SplitConformal(alpha=0.1, key_fn=season_key).fit(
        pd.DataFrame({"residual": resid, "month": built["month"]}))
    save_model_artifact(d / "runs" / "delhi_ncr_modelA_20261026T000000Z",
                        model, conf, fe_cols, GROUPS, meta={"city": "delhi_ncr"})
    return d / "runs"


@pytest.fixture
def statics() -> pd.DataFrame:
    return pd.DataFrame({
        "cell_id": CELL_IDS,
        "elevation_mean_m": [ELEV[c] for c in CELL_IDS],
        "elevation_std_m": 5.0,
        "elevation_min_m": [ELEV[c] - 4.0 for c in CELL_IDS],
        "elevation_max_m": [ELEV[c] + 4.0 for c in CELL_IDS],
        "water_fraction": 0.01,
    })


def _no2_fn(states=None, drop=None):
    states = states or {CELL_IDS[0]: ("observed", 42.0),
                        CELL_IDS[1]: ("insufficient", None),
                        CELL_IDS[2]: ("no_overpass", None),
                        CELL_IDS[3]: ("observed", 50.0)}

    def fn(city_cfg, date, cell_ids):
        rows = []
        for cid in cell_ids:
            if drop and cid in drop:
                continue
            st, val = states.get(cid, ("unavailable", None))
            rows.append({"cell_id": cid, "date": date,
                         "value_umol_m2": val, "state": st,
                         "reason_code": None if st == "observed" else "low_coverage",
                         "n_valid_fraction": 0.8 if st == "observed" else 0.1,
                         "subpixel_spread_umol_m2": 1.0,
                         "n_orbit_overlap": 1.0})
        return rows

    return fn


def _weather_fn(city_cfg, date, cell_ids):
    return [dict(
        cell_id=cid, date=date,
        wind_u10_inst=1.0, wind_v10_inst=1.0,
        wind_u10_l24=1.0, wind_v10_l24=1.0,
        wind_u10_l48=1.0, wind_v10_l48=1.0,
        wind_u100_inst=2.0, wind_v100_inst=2.0,
        wind_u100_l24=2.0, wind_v100_l24=2.0,
        wind_u100_l48=2.0, wind_v100_l48=2.0,
        t2m_k_inst=300.0, d2m_k_inst=292.0, sp_pa_inst=100_000.0,
        blh_m_inst=900.0, precip_l24_m=0.0, precip_l48_m=0.0,
    ) for cid in cell_ids]


def make_predictor(runs_dir, statics, states=None, drop=None) -> LivePredictor:
    art, name = load_latest_artifact(runs_dir, "delhi_ncr")
    return LivePredictor(
        city="delhi_ncr",
        city_cfg={"lon_min": 76.38, "lon_max": 78.62,
                  "lat_min": 27.60, "lat_max": 29.32},
        artifact=art, statics=statics, run_name=name,
        no2_fn=_no2_fn(states, drop), weather_fn=_weather_fn,
        version_fn=lambda cfg, date: ["2.8.0"])


@pytest.fixture
def predictor(runs_dir, statics) -> LivePredictor:
    return make_predictor(runs_dir, statics)


class TestCellLabels:
    def test_observed_cells_report_the_observation(self, predictor):
        out = predictor.compute(POLY, "2024-07-01")
        cells = {c["cell_id"]: c for c in out["cells"]}
        assert cells[CELL_IDS[0]]["source"] == "observed"
        assert cells[CELL_IDS[0]]["value"] == pytest.approx(42.0)
        assert cells[CELL_IDS[0]]["value_observed"] == pytest.approx(42.0)
        assert cells[CELL_IDS[3]]["value"] == pytest.approx(50.0)

    def test_non_observed_cells_are_reconstructed_from_the_model(self, predictor):
        out = predictor.compute(POLY, "2024-07-01")
        cells = {c["cell_id"]: c for c in out["cells"]}
        for cid in (CELL_IDS[1], CELL_IDS[2]):  # insufficient, no_overpass
            c = cells[cid]
            assert c["source"] == "reconstructed"
            assert c["value_observed"] is None
            assert c["value"] == pytest.approx(c["pred"])
            assert c["interval_lo"] < c["pred"] < c["interval_hi"]

    def test_d6_states_propagate_unchanged(self, predictor):
        out = predictor.compute(POLY, "2024-07-01")
        states = {c["cell_id"]: c["state"] for c in out["cells"]}
        assert states[CELL_IDS[0]] == "observed"
        assert states[CELL_IDS[1]] == "insufficient"
        assert states[CELL_IDS[2]] == "no_overpass"


class TestGeometryAndSummary:
    def test_exact_cell_set_and_unit_weights(self, predictor):
        out = predictor.compute(POLY, "2024-07-01")
        assert out["summary"]["n_cells"] == 4
        assert sorted(c["cell_id"] for c in out["cells"]) == sorted(CELL_IDS)
        assert all(c["weight"] == pytest.approx(1.0) for c in out["cells"])

    def test_weighted_mean_matches_cell_values(self, predictor):
        out = predictor.compute(POLY, "2024-07-01")
        vals = [c["value"] for c in out["cells"]]
        assert out["summary"]["weighted_mean"] == pytest.approx(sum(vals) / 4)
        assert out["summary"]["observed_fraction"] == pytest.approx(0.5)

    def test_edge_cells_keep_fractional_weight(self, predictor):
        # half a cell wide: the polygon covers cells at 77.00-77.05 partially
        half = [(77.02, 28.02), (77.04, 28.02), (77.04, 28.04), (77.02, 28.04)]
        out = predictor.compute(half, "2024-07-01")
        assert out["summary"]["n_cells"] == 1
        w = out["cells"][0]["weight"]
        assert 0.0 < w < 1.0
        # summary weights edge cells, so a fractional weight cannot inflate it
        assert out["summary"]["total_weight"] == pytest.approx(w)

    def test_sub_cell_polygon_gets_resolution_warning(self, predictor):
        tiny = [(77.01, 28.01), (77.02, 28.01), (77.02, 28.02), (77.01, 28.02)]
        out = predictor.compute(tiny, "2024-07-01")
        assert out["summary"]["n_cells"] == 1
        assert any("smaller than one grid cell" in w for w in out["warnings"])


class TestProvenance:
    def test_model_and_data_versions_are_reported(self, predictor):
        out = predictor.compute(POLY, "2024-07-01")
        assert out["model"]["run"] == "delhi_ncr_modelA_20261026T000000Z"
        assert out["model"]["alpha"] == pytest.approx(0.1)
        assert out["data"]["processor_versions"] == ["2.8.0"]
        assert out["date"] == "2024-07-01"
        assert out["grid_cell_deg"] == pytest.approx(0.05)

    def test_missing_fetch_rows_are_labelled_not_fabricated(self, runs_dir, statics):
        p = make_predictor(runs_dir, statics, drop={CELL_IDS[1]})
        out = p.compute(POLY, "2024-07-01")
        c = {c["cell_id"]: c for c in out["cells"]}[CELL_IDS[1]]
        assert c["state"] == "unavailable"
        assert c["reason_code"] == "fetch_missing"
        assert c["value_observed"] is None


class TestLoaders:
    def test_date_cap_is_d4_lag(self):
        assert latest_available_date(_date(2026, 10, 26)) == "2026-10-20"

    def test_latest_artifact_prefers_newest_complete_run(self, runs_dir, tmp_path):
        newer = runs_dir / "delhi_ncr_modelA_20270101T000000Z"
        newer.mkdir()
        (newer / "artifact.json").write_text("{}", encoding="utf-8")  # incomplete
        art, name = load_latest_artifact(runs_dir, "delhi_ncr")
        assert name == "delhi_ncr_modelA_20261026T000000Z"

    def test_no_artifact_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_latest_artifact(tmp_path, "delhi_ncr")

    def test_schema_mismatch_refused_at_load(self, runs_dir, monkeypatch):
        import no2map.live as live_mod
        monkeypatch.setattr(live_mod, "FEATURE_SCHEMA_VERSION", "fm-v999")
        with pytest.raises(RuntimeError, match="retrain"):
            load_latest_artifact(runs_dir, "delhi_ncr")

    def test_statics_loader_requires_cache(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_statics(tmp_path, "delhi_ncr")
