"""Tests for split-conformal intervals.

Empirical coverage on seeded fresh samples is the core claim: with enough
calibration points the interval covers at (or slightly above) 1-alpha on
independent data - distribution-free. The stratification ladder, the
small-n/max fallback, and the not-fitted guards are also pinned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from no2map.conformal import SplitConformal, conformal_quantile as cq


class TestQuantile:
    def test_exact_order_statistic(self):
        r = np.arange(1.0, 11.0)
        # k = ceil(11 * 0.9) = 10 -> sorted[9] = 10.0
        assert cq(r, alpha=0.1) == pytest.approx(10.0)
        # alpha = 0.2 -> k = ceil(11 * .8) = 9 -> 9.0
        assert cq(r, alpha=0.2) == pytest.approx(9.0)

    def test_small_n_falls_back_to_max(self):
        r = np.array([1.0, 2.0])
        assert cq(r, alpha=0.05) == pytest.approx(2.0)  # k = 2 > n

    def test_rejects_bad_alpha_and_empty(self):
        with pytest.raises(ValueError):
            cq(np.array([1.0]), alpha=0.0)
        with pytest.raises(ValueError):
            cq(np.array([], dtype=float), alpha=0.1)


class TestCoverage:
    def test_empirical_coverage_on_fresh_noise(self):
        rng = np.random.default_rng(11)
        calib = pd.DataFrame({"residual": rng.normal(size=400)})
        conf = SplitConformal(alpha=0.1, key_fn=None).fit(calib)
        fresh = pd.DataFrame({
            "pred": np.zeros(20_000),
            "truth": rng.normal(scale=1.0, size=20_000),
        })
        cov = conf.coverage(fresh, "pred", "truth")
        assert 0.87 < cov < 0.93

    def test_intervals_symmetric_and_tight_enough(self):
        rng = np.random.default_rng(3)
        conf = SplitConformal(alpha=0.5, key_fn=None).fit(
            pd.DataFrame({"residual": rng.normal(size=500)}))
        m = conf.margin(pd.DataFrame({"x": range(5)}))
        # median |N(0,1)| is ~0.674: 50%-level margin should sit near it
        assert 0.55 < np.median(m) < 0.80
        assert np.all(m > 0)


class TestStratification:
    def _frame(self, rng, n_per_stratum):
        rows = []
        for month in (1, 7):
            # winter (1) tighter residuals, summer (7) wider: different q
            scale = 1.0 if month == 1 else 3.0
            for resid in rng.normal(scale=scale, size=n_per_stratum):
                rows.append(dict(month=month, residual=resid))
        return pd.DataFrame(rows)

    def test_strata_get_different_quantiles(self):
        rng = np.random.default_rng(5)
        df = self._frame(rng, 300)
        conf = SplitConformal(
            alpha=0.1, key_fn=lambda row: int(row["month"])).fit(df)
        assert conf.q_by_stratum[1] < conf.q_by_stratum[7]

    def test_unseen_stratum_falls_back_to_pooled(self):
        rng = np.random.default_rng(5)
        df = self._frame(rng, 300)
        conf = SplitConformal(
            alpha=0.1, key_fn=lambda row: int(row["month"])).fit(df)
        unseen = pd.DataFrame([{"month": 12, "residual": 0.0}])
        pooled = conf.q_by_stratum[None]
        assert conf.margin(unseen)[0] == pytest.approx(pooled)

    def test_step_function_keys(self):
        # seasonal strata: month 1 -> winter key; 4 & 7 -> summer key;
        # month 3 maps to a THIRD key that calibration never saw -> pooled.
        # NOTE: int() is load-bearing - inside df.apply(axis=1) pandas upcasts
        # int columns to float when the frame is mixed-dtype, so str()-based
        # keys silently collapse to a single stratum (caught the hard way).
        df = pd.DataFrame({"month": [1, 4, 7] * 50,
                           "residual": np.abs(np.random.default_rng(0).normal(size=150))})
        conf = SplitConformal(
            alpha=0.2,
            key_fn=lambda row: {1: "W", 4: "X", 7: "X"}.get(int(row["month"]), "Q"),
        ).fit(df)
        assert set(k for k in conf.q_by_stratum if k is not None) == {"W", "X"}
        m = conf.margin(pd.DataFrame({"month": [1, 4, 3]}))
        assert m[0] == pytest.approx(conf.q_by_stratum["W"])
        assert m[1] == pytest.approx(conf.q_by_stratum["X"])
        assert m[2] == pytest.approx(conf.q_by_stratum[None])  # unseen key 'Q'


class TestGuards:
    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError, match="not calibrated"):
            SplitConformal(alpha=0.1).margin(pd.DataFrame([{"x": 1}]))

    def test_fit_requires_column(self):
        with pytest.raises(KeyError):
            SplitConformal(alpha=0.1).fit(pd.DataFrame([{"nope": 1}]))
