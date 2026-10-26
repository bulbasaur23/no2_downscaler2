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
from no2map.baselines import Climatology
from no2map.conformal import SplitConformal
from no2map.features import (FEATURE_GROUPS, TARGET, build_features,
                             feature_block)
from no2map.grid import cell_from_point
from no2map.live import (LivePredictor, latest_available_date,
                         load_g3_summary, load_latest_artifact, load_statics)
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
                    wind_u100_inst=rng.normal(2, 0.5), wind_v100_inst=rng.normal(2, 0.5),
                    wind_u100_l24=rng.normal(2, 0.5), wind_v100_l24=rng.normal(2, 0.5),
                    wind_u100_l48=rng.normal(2, 0.5), wind_v100_l48=rng.normal(2, 0.5),
                    t2m_k_inst=300.0 + rng.normal(0, 0.5),
                    d2m_k_inst=292.0 + rng.normal(0, 0.5),
                    sp_pa_inst=100_000.0, blh_m_inst=900.0 + rng.normal(0, 50),
                    precip_l24_m=0.0, precip_l48_m=0.0,
                ))
    return pd.DataFrame(rows)


def build_test_artifact(runs_dir: Path) -> Path:
    """Train a tiny artifact on the synthetic frame and save the bundle."""
    df = _training_frame()
    built = build_features(df)
    fe_cols = feature_block(built, GROUPS)
    model = XGBConfig(n_estimators=40).make(seed=0)
    model.fit(built[fe_cols], built[TARGET])
    resid = built[TARGET] - model.predict(built[fe_cols])
    conf = SplitConformal(alpha=0.1, key_fn=season_key).fit(
        pd.DataFrame({"residual": resid, "month": built["month"]}))
    clim = Climatology().fit(built)
    return save_model_artifact(
        runs_dir / "delhi_ncr_modelA_20261026T000000Z",
        model, conf, fe_cols, GROUPS,
        meta={"city": "delhi_ncr"}, climatology=clim)


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("live")
    build_test_artifact(d / "runs")
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

    def test_in_sample_request_is_flagged(self, runs_dir, statics):
        # a date inside the training window gives in-sample reconstructions;
        # the UI must say so rather than implying out-of-sample skill
        art, name = load_latest_artifact(runs_dir, "delhi_ncr")
        art.meta["months"] = ["2024-06", "2024-07"]
        p = LivePredictor(
            city="delhi_ncr",
            city_cfg={"lon_min": 76.38, "lon_max": 78.62,
                      "lat_min": 27.60, "lat_max": 29.32},
            artifact=art, statics=statics, run_name=name,
            no2_fn=_no2_fn(), weather_fn=_weather_fn,
            version_fn=lambda cfg, date: ["2.8.0"])
        assert any("training window" in w for w in
                   p.compute(POLY, "2024-07-01")["warnings"])
        art.meta["months"] = ["2024-01"]
        assert not any("training window" in w for w in
                       p.compute(POLY, "2024-07-01")["warnings"])


class TestParity:
    def test_assemble_matches_training_feature_build(self, runs_dir, statics):
        """G4 parity hook: identical raw inputs must produce identical features
        through the live assembly and through a straight training build."""
        from no2map.weather import WEATHER_COLUMNS

        df = _training_frame()
        day = "2024-06-01"
        train_rows = df[df["date"] == day].copy()
        ids = sorted(train_rows["cell_id"].unique())
        no2_rows = [{
            "cell_id": r.cell_id, "date": day,
            "value_umol_m2": r.value_umol_m2, "state": "observed",
            "reason_code": None, "n_valid_fraction": 1.0,
            "subpixel_spread_umol_m2": 0.0, "n_orbit_overlap": 1.0,
        } for r in train_rows.itertuples()]
        wx_rows = train_rows[["cell_id", "date", *WEATHER_COLUMNS]].to_dict("records")

        art, name = load_latest_artifact(runs_dir, "delhi_ncr")
        p = LivePredictor(
            city="delhi_ncr",
            city_cfg={"lon_min": 76.38, "lon_max": 78.62,
                      "lat_min": 27.60, "lat_max": 29.32},
            artifact=art, statics=statics, run_name=name,
            no2_fn=lambda cfg, d, wanted: [r for r in no2_rows
                                           if r["cell_id"] in wanted],
            weather_fn=lambda cfg, d, wanted: [r for r in wx_rows
                                               if r["cell_id"] in wanted],
            version_fn=lambda cfg, date: ["2.8.0"])

        live = build_features(p.assemble(day, ids))
        train = build_features(train_rows)
        cols = feature_block(live, GROUPS)
        a = live.sort_values("cell_id").reset_index(drop=True)[cols]
        b = train.sort_values("cell_id").reset_index(drop=True)[cols]
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


class TestNoObservationDay:
    def test_all_reconstructed_day_is_flagged(self, runs_dir, statics):
        # a genuine no-overpass day returns no observed cells: the response
        # must say the whole map is model output, not let it read as data
        states = {cid: ("no_overpass", None) for cid in CELL_IDS}
        p = make_predictor(runs_dir, statics, states)
        out = p.compute(POLY, "2024-07-01")
        assert out["summary"]["observed_fraction"] == 0.0
        assert all(c["source"] == "reconstructed" for c in out["cells"])
        assert any("every cell shown is model output" in w
                   for w in out["warnings"])

    def test_mostly_observed_day_is_not_flagged(self, predictor):
        out = predictor.compute(POLY, "2024-07-01")
        assert out["summary"]["observed_fraction"] > 0.0
        assert not any("every cell shown is model output" in w
                       for w in out["warnings"])


class TestClimatologyAnchor:
    """G3 returned `honest_report`: reconstructions must be shown beside the
    per-cell climatology they are anchored to, never alone."""

    def test_cells_carry_the_training_climatology(self, predictor):
        out = predictor.compute(POLY, "2024-07-01")
        built = build_features(_training_frame())
        clim = Climatology().fit(built)
        table = clim.by_cell_month.set_index(["cell_id", "month"])["value"]
        for c in out["cells"]:
            assert c["climatology"] == pytest.approx(
                float(table.loc[(c["cell_id"], 7)]), abs=1e-4)
        assert out["climatology_anchor"]["available"] is True

    def test_summary_reports_the_anchor_as_a_weighted_mean(self, predictor):
        out = predictor.compute(POLY, "2024-07-01")
        vals = [c["climatology"] for c in out["cells"]]
        assert out["summary"]["weighted_mean_climatology"] == pytest.approx(
            sum(vals) / len(vals), abs=1e-3)

    def test_g3_block_is_passed_through_to_the_response(self, predictor):
        predictor.g3 = {"g3_mode": "honest_report",
                        "blocked": {"beats_beyond_noise": False}}
        out = predictor.compute(POLY, "2024-07-01")
        assert out["g3"]["g3_mode"] == "honest_report"
        assert out["g3"]["blocked"]["beats_beyond_noise"] is False


class TestG3SummaryLoader:
    def test_missing_report_is_none(self, tmp_path):
        assert load_g3_summary(tmp_path, "delhi_ncr") is None

    def test_reads_the_verdict_and_both_protocols(self, tmp_path):
        import json
        rep = {
            "generated_at": "2026-10-26T00:00:00+00:00",
            "blocked": {"seed_delta": {"delta_mean": 2.03, "delta_sd": 0.2}},
            "chronological": {"seed_delta": {"delta_mean": -1.58, "delta_sd": 0.02}},
            "verdict": {"g3_mode": "honest_report",
                        "beats_blocked_beyond_noise": False,
                        "beats_chronological_beyond_noise": True,
                        "required_actions": ["label climatology-anchored estimates"],
                        "note": "no tuning on the test period"},
        }
        (tmp_path / "g3_report_delhi_ncr.json").write_text(
            json.dumps(rep), encoding="utf-8")
        out = load_g3_summary(tmp_path, "delhi_ncr")
        assert out["g3_mode"] == "honest_report"
        assert out["blocked"]["model_minus_climatology_rmse"] == pytest.approx(2.03)
        assert out["blocked"]["beats_beyond_noise"] is False
        assert out["chronological"]["beats_beyond_noise"] is True
        assert out["required_actions"]


class TestCache:
    def test_second_identical_request_skips_the_fetch(self, runs_dir, statics,
                                                      tmp_path):
        from no2map.cache import CellCache
        calls = {"n": 0}
        base_no2 = _no2_fn()

        def counting_no2(city_cfg, date, cell_ids):
            calls["n"] += 1
            return base_no2(city_cfg, date, cell_ids)

        art, name = load_latest_artifact(runs_dir, "delhi_ncr")
        p = LivePredictor(
            city="delhi_ncr",
            city_cfg={"lon_min": 76.38, "lon_max": 78.62,
                      "lat_min": 27.60, "lat_max": 29.32},
            artifact=art, statics=statics, run_name=name,
            cache=CellCache(tmp_path / "c.sqlite"),
            no2_fn=counting_no2, weather_fn=_weather_fn,
            version_fn=lambda cfg, date: ["2.8.0"])

        first = p.compute(POLY, "2024-07-01")
        second = p.compute(POLY, "2024-07-01")
        assert first["cache_hit"] is False and second["cache_hit"] is True
        assert calls["n"] == 1  # full hit: no Earth Engine work at all
        assert first["cells"] == second["cells"]
        assert first["summary"] == second["summary"]

    def test_different_date_is_a_miss(self, runs_dir, statics, tmp_path):
        from no2map.cache import CellCache
        p = make_predictor(runs_dir, statics)
        p.cache = CellCache(tmp_path / "c.sqlite")
        assert p.compute(POLY, "2024-07-01")["cache_hit"] is False
        assert p.compute(POLY, "2024-07-02")["cache_hit"] is False

    def test_new_model_version_invalidates(self, runs_dir, statics, tmp_path):
        from no2map.cache import CellCache
        p = make_predictor(runs_dir, statics)
        p.cache = CellCache(tmp_path / "c.sqlite")
        p.compute(POLY, "2024-07-01")
        p.run_name = "retrained_run_b"
        assert p.compute(POLY, "2024-07-01")["cache_hit"] is False


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
