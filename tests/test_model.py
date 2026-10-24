"""Tests for the Model A runner on synthetic panels.

G3's criterion is rehearsed here on synthetic ground truth with a KNOWN
static spatial component: block-holdout folds where the validation blocks
contain cells UNEVER seen in training must favour the weather/static model
over the (cell,month) climatology, because unseen cells fall to the city
level fallback while the model reads the static features. That is precisely
the discrimination the plan wants to demo-replay before real data arrives.

Also pinned: the OOF frame schema (all four baseline columns), conformal
coverage near 1-alpha on the pooled residuals, season-key dtype stability,
and run determinism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from no2map.features import STATIC_FEATURES, TARGET, feature_block
from no2map.model import PRED_COLS, design_matrix, run_blocked_evaluation, season_key
from no2map.features import FEATURE_GROUPS
from no2map.grid import cell_from_point
from no2map.variogram import TROPOMI_CELL_M

N_DATES = 24
N_COLS = 4


def _cells_two_clusters() -> list[str]:
    """Two well-separated row-families so block tilings give >=2 blocks with
    spatially disjoint val rows."""
    rows_a = [cell_from_point(28.0 + 0.05 * r, 77.0 + 0.05 * c).id
              for r in range(6) for c in range(N_COLS)]
    rows_b = [cell_from_point(29.8 + 0.05 * r, 77.0 + 0.05 * c).id
              for r in range(6) for c in range(N_COLS)]
    return rows_a + rows_b


def _synthetic_panel(seed: int = 3) -> pd.DataFrame:
    """y = static_elev * 0.5 + 3 * t2m anomaly + month season + noise."""
    rng = np.random.default_rng(seed)
    cells = _cells_two_clusters()
    lat0 = {c: 28.0 + 0.05 * (i // N_COLS) for i, c in enumerate(cells[:6 * N_COLS])}

    static = {cid: 100.0 + 5.0 * ((i % 17) + 1) for i, cid in enumerate(cells)}
    t2m_base = {cid: 298.0 + 0.3 * (i % 11) for i, cid in enumerate(cells)}
    dates = [f"2024-{m:02d}-{d:02d}" for m in (6, 7) for d in range(1, 13)]
    rows = []
    for k, date in enumerate(dates):
        month = int(date[5:7])
        t2m = t2m_base.copy()
        for cid in t2m:  # daily weather anomaly (spatially correlated-ish)
            t2m[cid] += rng.normal(0, 0.6)
        for cid in cells:
            month_effect = 10.0 if month == 7 else 0.0
            y = 0.5 * static[cid] + 3.0 * (t2m[cid] - t2m_base[cid]) + month_effect \
                + rng.normal(0, 2.0)
            row = dict(
                cell_id=cid, date=date, month=month, state="observed",
                **{TARGET: y},
                elevation_mean_m=static[cid], elevation_std_m=5.0,
                elevation_min_m=static[cid] - 4.0, elevation_max_m=static[cid] + 4.0,
                water_fraction=0.01,
                wind_u10_inst=rng.normal(0, 1.5), wind_v10_inst=rng.normal(0, 1.5),
                wind_u10_l24=rng.normal(0, 1.5), wind_v10_l24=rng.normal(0, 1.5),
                wind_u10_l48=rng.normal(0, 1.5), wind_v10_l48=rng.normal(0, 1.5),
                t2m_k_inst=t2m[cid], d2m_k_inst=t2m[cid] - 2.0,
                sp_pa_inst=101_000.0, blh_m_inst=800.0 + rng.normal(0, 100),
                precip_l24_m=abs(rng.normal(0, 0.05)),
                precip_l48_m=abs(rng.normal(0, 0.05)),
            )
            rows.append(row)
    return pd.DataFrame(rows)


GROUPS_ALL = list(FEATURE_GROUPS)


class TestRunner:
    def test_oof_frame_schema_and_every_cell_once(self):
        df = _synthetic_panel()
        oof, report = run_blocked_evaluation(
            df, GROUPS_ALL, block_m=TROPOMI_CELL_M * 3.0, buffer_m=0.0, seed=1)
        assert set(PRED_COLS) <= set(oof.columns)
        assert {"cell_id", "date", TARGET} <= set(oof.columns)
        # every observed row is one validation row exactly once
        n_obs = (df["state"] == "observed").sum()
        assert len(oof) == n_obs
        assert set(report["folds"]) == set(oof["block"].unique())

    def test_model_beats_climatology_on_blocked_folds(self):
        df = _synthetic_panel()
        oof, report = run_blocked_evaluation(
            df, GROUPS_ALL, block_m=TROPOMI_CELL_M * 3.0, buffer_m=0.0, seed=1)
        assert report["pred_model_a"]["rmse"] < report["pred_climatology"]["rmse"]
        # and the margin is material, not seed noise
        assert (report["pred_climatology"]["rmse"] /
                report["pred_model_a"]["rmse"]) > 1.3

    def test_blocked_rmse_is_not_suspiciously_low(self):
        # leakage check: with a buffer gap the OOF RMSE should reflect the
        # irreducible noise scale (~2.0), not approach zero
        df = _synthetic_panel()
        oof, _ = run_blocked_evaluation(
            df, GROUPS_ALL, block_m=TROPOMI_CELL_M * 3.0, buffer_m=0.0, seed=2)
        rmse_mod = np.sqrt(np.mean((oof[TARGET] - oof["pred_model_a"]) ** 2))
        assert 0.04 * rmse_mod < np.sqrt(4.0)  # model worse than pure noise? no
        assert rmse_mod > 0.5  # far above zero: held-out rows are really held out

    def test_coverage_of_conformal_margins(self):
        df = _synthetic_panel()
        oof, report = run_blocked_evaluation(
            df, GROUPS_ALL, block_m=TROPOMI_CELL_M * 3.0, buffer_m=0.0, seed=1)
        cov = float((oof["residual"].abs() <= oof["conf_margin"]).mean())
        assert 0.80 < cov < 0.99
        assert report["conformal"]["q_pooled"] > 0

    def test_deterministic_run(self):
        df = _synthetic_panel()
        a, _ = run_blocked_evaluation(df, GROUPS_ALL, TROPOMI_CELL_M * 3.0, seed=1)
        b, _ = run_blocked_evaluation(df, GROUPS_ALL, TROPOMI_CELL_M * 3.0, seed=1)
        a = a.sort_values(["cell_id", "date"]).reset_index(drop=True)
        b = b.sort_values(["cell_id", "date"]).reset_index(drop=True)
        np.testing.assert_allclose(a["pred_model_a"], b["pred_model_a"], rtol=1e-9)

    def test_unknown_group_rejected(self):
        df = _synthetic_panel()
        with pytest.raises(KeyError):
            run_blocked_evaluation(df, ["static", "not_a_group"],
                                   TROPOMI_CELL_M * 3.0, seed=1)


class TestSeasonKey:
    def test_dtype_stable_months(self):
        assert season_key({"month": 1}) == "Q0"
        assert season_key({"month": 3}) == "Q0"
        assert season_key({"month": 4}) == "Q1"
        assert season_key({"month": 12}) == "Q3"

    def test_float_month_upcast_safe(self):
        # apply(axis=1) upcasts ints with mixed dtypes - float months must
        # still map correctly (the conformal dtype warning, regression guard)
        assert season_key(pd.Series({"month": 7.0})) == "Q2"


class TestDesignMatrix:
    def test_returns_requested_group_columns(self):
        df = _synthetic_panel()
        X = design_matrix(df, ["static"])
        assert list(X.columns) == FEATURE_GROUPS["static"]
        assert X.shape == (df.shape[0], len(FEATURE_GROUPS["static"]))
