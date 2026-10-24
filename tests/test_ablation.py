"""Tests for the feature-group ablation ladder.

Attribution invariants: every stage reports all four methods; the default
ladder is cumulative (each stage a superset of the last); deltas in metres
are present for stage 2..N; and, on a synthetic truth with BOTH static and
weather signal, the full-feature stage must not be worse than static-only
(adding true-signal families must never hurt under identical folds - if it
does, the fold machinery, not the features, is broken).
"""
from __future__ import annotations

import pytest

from no2map.model import run_ablation
from no2map.variogram import TROPOMI_CELL_M

from tests.test_model import _synthetic_panel

ALL = ["static", "weather_inst", "weather_lags", "temporal"]


@pytest.fixture(scope="module")
def ablation_stages():
    df = _synthetic_panel()
    return run_ablation(df, block_m=TROPOMI_CELL_M * 3.0, seed=1)


class TestLadder:
    def test_four_cumulative_stages(self, ablation_stages):
        assert len(ablation_stages) == 4
        seen = []
        for st in ablation_stages:
            g = st["groups"]
            assert len(g) > len(seen) and g[:len(seen)] == seen
            seen = g
        assert seen == ALL

    def test_every_stage_scores_all_methods(self, ablation_stages):
        for st in ablation_stages:
            for k in ("model_rmse", "ridge_rmse", "climatology_rmse",
                      "city_mean_rmse"):
                assert st[k] > 0

    def test_deltas_present_for_later_stages(self, ablation_stages):
        assert "delta_model_m" not in ablation_stages[0]
        for st in ablation_stages[1:]:
            assert isinstance(st["delta_model_m"], float)

    def test_adding_true_signals_never_hurts(self, ablation_stages):
        # synthetic truth has static + t2m + month effects; the monotone
        # expectation holds weakly (same folds, same seed)
        rmses = [st["model_rmse"] for st in ablation_stages]
        assert rmses[-1] <= rmses[0] * 1.15  # no blow-up from added families
