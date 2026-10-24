"""Tests for Phase 2 baselines (the honesty anchors).

Each baseline is checked against hand-computable synthetic data, including
the fallback ladders that the live polygon path depends on, the
train-only-fitting integrity rule, and ridge mathematics (standardization
invariance, unpenalized intercept, shrinkage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from no2map.baselines import Climatology, DailyCityMean, RidgeBaseline
from no2map.features import STATIC_FEATURES, TARGET


def _frame(n_cells: int = 4, n_days: int = 4, **fixed) -> pd.DataFrame:
    """Synthetic observed frame with deterministic spatial+seasonal signal:
    value = 10*cell_index + 5*month + noise-free baseline."""
    rows = []
    dates = [f"2024-{m:02d}-{d:02d}" for m in (6, 7) for d in (1, 2)]
    for c in range(n_cells):
        for d in dates:
            month = int(d[5:7])
            rows.append(dict(
                cell_id=f"cell_{c}", date=d, month=month,
                **{TARGET: 10.0 * c + 5.0 * month},
            ))
    df = pd.DataFrame(rows)
    for k, v in fixed.items():
        df[k] = v
    return df


class TestClimatology:
    def test_median_by_cell_month(self):
        df = _frame()
        clim = Climatology().fit(df)
        # cell_1, month 6: values all 40 -> median 40; month 7 -> 45
        got = clim.predict(pd.DataFrame([dict(cell_id="cell_1", month=6)]))
        assert got.iloc[0] == pytest.approx(40.0)
        got = clim.predict(pd.DataFrame([dict(cell_id="cell_1", month=7)]))
        assert got.iloc[0] == pytest.approx(45.0)

    def test_fallback_ladder_cell_median(self):
        df = _frame()
        clim = Climatology().fit(df)
        # (cell_2, month 12): no pair row -> cell_2 median of {50,50,55,55}
        got = clim.predict(pd.DataFrame([dict(cell_id="cell_2", month=12)]))
        assert got.iloc[0] == pytest.approx(52.5)

    def test_fallback_ladder_month_median(self):
        df = _frame()
        clim = Climatology().fit(df)
        # unseen cell -> month 6 city median across cells = median{30,40,50,60}
        got = clim.predict(pd.DataFrame([dict(cell_id="never_seen", month=6)]))
        assert got.iloc[0] == pytest.approx(45.0)

    def test_fallback_ladder_city_median(self):
        df = _frame()
        clim = Climatology().fit(df)
        got = clim.predict(pd.DataFrame([dict(cell_id="never_seen", month=13)]))
        assert got.iloc[0] == pytest.approx(clim.city_median)
        assert np.isfinite(clim.city_median)

    def test_fallback_rate_reports_unseen_pairs(self):
        clim = Climatology().fit(_frame())
        eval_df = pd.DataFrame([
            dict(cell_id="cell_0", month=6),   # seen pair
            dict(cell_id="zzz", month=6),      # unseen cell
        ])
        assert clim.fallback_rate(eval_df) == pytest.approx(0.5)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            Climatology().predict(pd.DataFrame([dict(cell_id="a", month=1)]))

    def test_predict_preserves_index_and_name(self):
        idx = pd.Index([7, 3], name="rows")
        clim = Climatology().fit(_frame())
        out = clim.predict(pd.DataFrame(
            [dict(cell_id="cell_0", month=6)] * 2, index=idx))
        assert out.index.equals(idx)
        assert out.name == "pred_climatology"


class TestDailyCityMean:
    def test_daily_mean_is_spatial_mean(self):
        df = _frame()
        day1 = df[df["date"] == "2024-06-01"]
        expected = day1[TARGET].mean()
        dcm = DailyCityMean().fit(df)
        got = dcm.predict(pd.DataFrame([dict(date="2024-06-01")]))
        assert got.iloc[0] == pytest.approx(expected)

    def test_unseen_date_falls_back_to_month_of_year_mean(self):
        dcm = DailyCityMean().fit(_frame())
        got = dcm.predict(pd.DataFrame([dict(date="2025-06-30")]))
        # June never seen on that date -> month-of-year mean from training
        assert got.iloc[0] == pytest.approx(dcm.by_month_of_year["06"])
        assert got.iloc[0] == pytest.approx(45.0)

    def test_unseen_month_falls_back_to_city_mean(self):
        # month 01 never seen in the fixture (months 6, 7) -> city mean = 47.5
        dcm = DailyCityMean().fit(_frame())
        got = dcm.predict(pd.DataFrame([dict(date="2025-01-01")]))
        assert got.iloc[0] == pytest.approx(dcm.city_mean)
        assert got.iloc[0] == pytest.approx(47.5)

    def test_month_of_year_uses_mean_of_daily_means(self):
        # two june dates with different per-day city means -> moy = mean of both
        df = _frame()
        df.loc[df["date"] == "2024-06-01", TARGET] -= 30.0  # day1 mean 15, day2 45
        dcm = DailyCityMean().fit(df)
        assert dcm.by_month_of_year["06"] == pytest.approx((15.0 + 45.0) / 2.0)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            DailyCityMean().predict(pd.DataFrame([dict(date="2024-01-01")]))


class TestRidgeMath:
    def test_recovers_exact_linear_relation(self):
        df = _frame()
        df["x1"] = df[TARGET] / 4.0  # y = 4*x1 exactly
        ridge = RidgeBaseline().fit(df, ["x1"], lam=1e-9)
        preds = ridge.predict(df)
        assert np.allclose(preds.to_numpy(), df[TARGET].to_numpy(), rtol=1e-6)

    def test_intercept_unpenalized_with_strong_lambda(self):
        # with lam -> inf and nonzero centered y, a penalized intercept would
        # collapse; the estimate must keep the mean structure: predictions at
        # train-x means equal train-y mean regardless of lambda
        df = _frame()
        df["x1"] = np.linspace(0, 10, len(df))
        df[TARGET] = 3.0 + 2.0 * df["x1"]
        ridge = RidgeBaseline().fit(df, ["x1"], lam=1e6)
        mean_x_pred = ridge.predict(pd.DataFrame(
            [dict(cell_id="cell_0", date="2024-06-01", month=6, x1=df["x1"].mean())]))
        assert mean_x_pred.iloc[0] == pytest.approx(df[TARGET].mean(), rel=1e-3)

    def test_shrinkage_monotone_in_lambda(self):
        df = _frame()
        df["x1"] = df[TARGET] / 4.0
        coef_lo = RidgeBaseline().fit(df, ["x1"], lam=1e-6).coef[0]
        coef_hi = RidgeBaseline().fit(df, ["x1"], lam=1e6).coef[0]
        assert 0 < abs(coef_hi) < abs(coef_lo)

    def test_standardization_scale_invariance(self):
        # ridge on standardized features: rescaling X must not change preds
        df = _frame()
        df["x1"] = df[TARGET] / 4.0
        df["x2"] = np.sin(np.linspace(0, 6.28, len(df)))
        r1 = RidgeBaseline().fit(df, ["x1", "x2"], lam=1.0)
        df_scaled = df.copy()
        df_scaled["x1"] = df_scaled["x1"] * 1000.0
        df_scaled["x2"] = df_scaled["x2"] * 0.001
        r2 = RidgeBaseline().fit(df_scaled, ["x1", "x2"], lam=1.0)
        assert np.allclose(r1.predict(df).to_numpy(),
                           r2.predict(df_scaled).to_numpy(), rtol=1e-9)

    def test_nan_imputation_uses_train_median(self):
        # NaNs at predict must be filled with the TRAIN median, so the
        # prediction equals the prediction for that column at its train median
        df = _frame()
        df["x1"] = df[TARGET] / 4.0
        ridge = RidgeBaseline().fit(df, ["x1"], lam=1e-9)
        probe = df.head(1).copy()
        probe["x1"] = np.nan
        probe_median = df.head(1).copy()
        probe_median["x1"] = ridge.medians[0]
        assert ridge.predict(probe).iloc[0] == pytest.approx(
            ridge.predict(probe_median).iloc[0])
        # and the stored median is the training median, unpolluted
        assert ridge.medians[0] == pytest.approx(float(df["x1"].median()))

    def test_empty_feature_list_rejected(self):
        with pytest.raises(ValueError, match="feature_cols"):
            RidgeBaseline().fit(_frame(), [])

    def test_static_only_role_confirmable(self):
        df = _frame()
        for c in STATIC_FEATURES:
            df[c] = float(len(df))
        ridge = RidgeBaseline().fit(df, STATIC_FEATURES, lam=1.0)
        assert ridge.is_static_only()

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            RidgeBaseline().predict(_frame())


class TestIntegrity:
    def test_all_baselines_read_only_target_constant(self):
        # all fitters must fail loudly if the target column is absent
        df = _frame().drop(columns=[TARGET])
        for fits in (
            lambda: Climatology().fit(df),
            lambda: DailyCityMean().fit(df),
        ):
            with pytest.raises(KeyError):
                fits()
