"""Tests for two-level scoring and yardstick metrics.

Pinned against hand-computable frames:
- level: tracks per-DATE city means only (dates without observed cells are
  excluded - a constant guard against silent pooling),
- pattern: within-day correlation means what it says; the classic two-model
  failure (identical levels, different per-cell structure) is reproduced and
  the pattern score must discriminate where level scores cannot,
- the score dict is complete for the Phase-3 runner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from no2map.evaluation import (
    bias,
    full_scores,
    level_scores,
    mae,
    pattern_scores,
    pearson,
    rmse,
    TARGET,
)


def _frame() -> pd.DataFrame:
    """4 dates x 3 cells. City level varies by date (1, 2, 4, 8); within a
    date cells have levels 1, 2, 3 relative."""
    rows = []
    for k, level in enumerate([1.0, 2.0, 4.0, 8.0]):
        for c in range(3):
            rows.append(dict(
                cell_id=f"cell_{c}",
                date=f"2024-06-{k + 1:02d}",
                **{TARGET: level * (c + 1)},
            ))
    return pd.DataFrame(rows)


class TestScalars:
    def test_rmse_mae_bias_hand_values(self):
        y = np.array([2.0, 4.0, 6.0])
        p = np.array([1.0, 4.0, 9.0])
        assert rmse(y, p) == pytest.approx(np.sqrt((1 + 0 + 9) / 3))
        assert mae(y, p) == pytest.approx((1 + 0 + 3) / 3)
        assert bias(y, p) == pytest.approx(((-1) + 0 + 3) / 3)


class TestPearson:
    def test_perfect_linear(self):
        x = np.linspace(1, 9, 20)
        assert pearson(x, 3 * x + 4) == pytest.approx(1.0)
        assert pearson(x, -x) == pytest.approx(-1.0)

    def test_constant_input_is_nan_not_crash(self):
        with pytest.raises(ValueError):
            pearson(np.array([1.0]), np.array([2.0]))
        assert np.isnan(pearson(np.ones(10), np.arange(10.0)))

    def test_anticorrelated(self):
        x = np.linspace(1, 9, 10)
        assert pearson(x, -2 * x + 1) == pytest.approx(-1.0)


class TestLevelScores:
    def test_tracks_daily_means(self):
        df = _frame()
        out = level_scores(df.assign(pred=df[TARGET]))
        assert out["r"] == pytest.approx(1.0)
        assert out["rmse"] == pytest.approx(0.0)
        assert out["n_dates"] == 4

    def test_level_rmse_computed_from_daily_means(self):
        df = _frame()
        # pred = obs + 0.5 daily shift: per-date means differ by exactly 0.5
        out = level_scores(df.assign(pred=df[TARGET] + 0.5))
        assert out["rmse"] == pytest.approx(0.5)

    def test_dates_without_obs_are_excluded(self):
        df = _frame()
        # observed target for 2024-06-03 removed entirely
        broken = df[~(df["date"] == "2024-06-03") | df[TARGET].isna()].copy()
        broken.loc[broken["date"] == "2024-06-03", TARGET] = np.nan  # keep pred row
        b = df[df.date != "2024-06-03"].copy()
        b.loc[b.date != "2024-06-03", "pred"] = b[TARGET]
        out = level_scores(b.assign(pred=b[TARGET]))
        assert out["n_dates"] == 3  # the all-NaN date carries no level info


class TestPatternScores:
    def test_identical_pred_beyond_level(self):
        df = _frame()
        out = pattern_scores(df.assign(pred=df[TARGET]), min_cells=2)
        assert out["mean_r"] == pytest.approx(1.0)

    def test_discriminates_structure_where_level_cannot(self):
        # Model-L: city mean plus per-cell NOISE (no real structure); the
        # level scores of the two models would be very close, but the pattern
        # score must separate them decisively.
        df = _frame()
        city_mean = df.groupby("date")[TARGET].transform("mean")
        rng = np.random.default_rng(0)
        slope = {0: 0.4, 1: 1.0, 2: 1.6}
        structured = city_mean * df["cell_id"].str[-1].astype(int).map(slope)
        model_l = df.assign(pred=city_mean + rng.normal(size=len(df)) * 3.0)
        model_p = df.assign(pred=structured)
        pat_l = pattern_scores(model_l, min_cells=2)
        pat_p = pattern_scores(model_p, min_cells=2)
        assert pat_l["mean_r"] < 0.35
        assert pat_p["mean_r"] > pat_l["mean_r"] + 0.45

    def test_min_cells_guard(self):
        df = _frame()
        two_cell_day = df[df["cell_id"].isin(["cell_0", "cell_1"])]
        out = pattern_scores(two_cell_day.assign(pred=two_cell_day[TARGET]),
                             min_cells=3)
        assert out["n_dates"] == 0
        assert np.isnan(out["mean_r"])


class TestFullScores:
    def test_complete_dict(self):
        df = _frame()
        s = full_scores(df.assign(pred=df[TARGET]))
        for k in ("n", "rmse", "mae", "bias",
                  "level_rmse", "level_r", "level_n_dates",
                  "pattern_mean_r", "pattern_n_dates"):
            assert k in s
        assert s["rmse"] == pytest.approx(0.0)
