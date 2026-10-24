"""Tests for the version-shift harness.

Rehearses the plan's processor-update protocol on synthetic panels whose
"shift" is a controlled level change (+8 umol/m2 on eval months plus a new
season): the frozen-report machinery must show the bias honestly (that is
the point of the shift set), while the premise guard rejects any overlap
between training and eval processor versions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from no2map.features import FEATURE_GROUPS, TARGET
from no2map.model import PRED_COLS
from no2map.version_shift import fit_all, predict_frozen, version_shift_report

from tests.test_model import _synthetic_panel  # reuse the seeded generator

GROUPS = list(FEATURE_GROUPS)


def _eval_frame(seed: int = 9, shift: float = 0.0, months=("2025-11", "2025-12")) -> pd.DataFrame:
    """Same generator family as the training panel, later months, optional
    level shift standing in for a processor-update distribution change."""
    df = _synthetic_panel(seed=seed)
    # move all dates into the eval months, keep within-month day spread
    repl = {}
    labels = sorted(set(df["date"]))
    for k, old in enumerate(labels):
        d = f"{months[k % len(months)]}-{(k % 27) + 1:02d}"
        repl[old] = d
    df["date"] = df["date"].map(repl)
    df["month"] = df["date"].str.slice(5, 7).astype(int)
    if shift:
        df[TARGET] = df[TARGET] + shift
    return df


TRAIN_VERSIONS = ["2.8.0"]


class TestPremiseGuard:
    def test_overlap_between_train_and_eval_versions_rejected(self):
        train = _synthetic_panel()
        ev = _eval_frame()
        with pytest.raises(ValueError, match="premise"):
            version_shift_report(
                train, {"2025-11": ev}, TRAIN_VERSIONS,
                {"2025-11": ["2.8.0"]}, GROUPS, seed=1)

    def test_new_version_runs(self):
        train = _synthetic_panel()
        ev = _eval_frame()
        rep = version_shift_report(
            train, {"2025-11": ev}, TRAIN_VERSIONS,
            {"2025-11": ["02.09.01"]}, GROUPS, seed=1)
        assert rep["per_month"]["2025-11"]["n_rows"] == len(ev)
        assert rep["eval_versions"] == {"2025-11": ["02.09.01"]}


class TestFrozenScoring:
    def test_shift_bias_is_reported_not_hidden(self):
        train = _synthetic_panel()
        ev = _eval_frame(shift=8.0)  # processor-update level change
        rep = version_shift_report(
            train, {"2025-11": ev}, TRAIN_VERSIONS,
            {"2025-11": ["02.09.01"]}, GROUPS, seed=1)
        s = rep["per_month"]["2025-11"]["scores"]
        for col in PRED_COLS:
            assert abs(s[col]["bias"]) > 4.0, (col, s[col]["bias"])

    def test_all_methods_scored_with_every_metric(self):
        train = _synthetic_panel()
        ev = _eval_frame()
        rep = version_shift_report(
            train, {"2025-11": ev}, TRAIN_VERSIONS,
            {"2025-11": ["02.09.01"]}, GROUPS, seed=1)
        s = rep["per_month"]["2025-11"]["scores"]
        for col in PRED_COLS:
            for k in ("rmse", "bias", "level_rmse", "level_r",
                      "pattern_mean_r", "coverage", "mean_width"):
                assert k in s[col], (col, k)
            assert 0.0 <= s[col]["coverage"] <= 1.0
            assert s[col]["mean_width"] > 0

    def test_coverage_on_unshifted_eval_near_nominal(self):
        # no shift: frozen intervals (pooled-season fallback for Q3) should
        # still roughly cover at least half their nominal level (alpha=0.1)
        train = _synthetic_panel()
        ev = _eval_frame()
        rep = version_shift_report(
            train, {"2025-11": ev}, TRAIN_VERSIONS,
            {"2025-11": ["02.09.01"]}, GROUPS, seed=1)
        s = rep["per_month"]["2025-11"]["scores"]["pred_model_a"]
        assert 0.5 <= s["coverage"] <= 1.0


class TestFitPredict:
    def test_predict_frozen_columns_and_rows(self):
        fitted = fit_all(_synthetic_panel(), GROUPS, seed=1, alpha=0.1)
        ev = _eval_frame()
        out = predict_frozen(fitted, ev, GROUPS)
        assert len(out) == len(ev)
        assert {c for c in PRED_COLS} <= set(out.columns)
        # frozen means deterministic: run twice, identical predictions
        out2 = predict_frozen(fitted, ev, GROUPS)
        np.testing.assert_allclose(out["pred_model_a"], out2["pred_model_a"])
