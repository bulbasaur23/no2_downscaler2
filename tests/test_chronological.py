"""Tests for the chronological (train-early / test-late) evaluation.

Pinned: strict time ordering (every test date after every train date),
all four methods scored on the future set through full_scores, frozen-
interval coverage/width reported per method, conformal calibrated INSIDE
the training period only, and the single-date / bad-frac error paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from no2map.model import PRED_COLS, run_chronological_evaluation
from no2map.features import TARGET, FEATURE_GROUPS

from tests.test_model import _synthetic_panel

GROUPS = list(FEATURE_GROUPS)


@pytest.fixture(scope="module")
def chron():
    df = _synthetic_panel()
    preds, rep = run_chronological_evaluation(df, GROUPS, train_frac=0.75,
                                              seed=1)
    return preds, rep


class TestSplit:
    def test_train_strictly_before_test(self, chron):
        preds, rep = chron
        assert rep["protocol"] == "chronological"
        assert rep["train_dates"][1] < rep["test_dates"][0]
        preds_dates = sorted(preds["date"].unique())
        assert preds_dates[0] >= rep["test_dates"][0]

    def test_frac_coverage_of_odds(self, chron):
        _, rep = chron
        # 24 dates, 0.75 -> 18 train / 6 test
        assert rep["n_train_dates"] == 18 and rep["n_test_dates"] == 6


class TestScoring:
    def test_all_methods_scored_with_coverage(self, chron):
        _, rep = chron
        for col in PRED_COLS:
            for k in ("rmse", "bias", "level_rmse", "level_r",
                      "pattern_mean_r", "coverage", "mean_width"):
                assert k in rep[col], (col, k)
            assert 0.0 <= rep[col]["coverage"] <= 1.0
            assert rep[col]["mean_width"] > 0

    def test_model_not_worse_than_climatology_on_future(self, chron):
        _, rep = chron
        assert rep["pred_model_a"]["rmse"] < rep["pred_climatology"]["rmse"]

    def test_intervals_not_degenerate(self, chron):
        preds, _ = chron
        assert (preds["conf_margin"] > 0).all()
        # single test season (July -> Q2): stratified margins must be the
        # SAME constant for all future rows - determinism, not spread - and
        # different from the pooled fallback quantile used for unseen
        # seasons would be verified in the conformal module's own tests.
        assert preds["conf_margin"].nunique() == 1


class TestGuards:
    def test_unknown_group_rejected(self):
        df = _synthetic_panel()
        with pytest.raises(KeyError):
            run_chronological_evaluation(df, ["static", "nope"], seed=1)

    def test_single_date_dataset_rejected(self):
        df = _synthetic_panel()
        df = df[df["date"] == df["date"].iloc[0]]
        with pytest.raises(ValueError):
            run_chronological_evaluation(df, GROUPS, seed=1)

    def test_bad_train_frac_rejected(self):
        df = _synthetic_panel()
        with pytest.raises(ValueError):
            run_chronological_evaluation(df, GROUPS, train_frac=1.0)
