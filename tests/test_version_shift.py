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
from no2map.version_shift import (fit_all, partition_eval_days,
                                  predict_frozen, version_shift_report)

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

    def test_coverage_ignores_rows_without_an_observation(self):
        # exported months contain insufficient/no_overpass rows with NaN
        # target: they are not interval misses and must not deflate coverage
        train = _synthetic_panel()
        ev = _eval_frame()
        half = len(ev) // 2
        ev = ev.copy()
        ev.loc[ev.index[:half], TARGET] = np.nan
        rep = version_shift_report(
            train, {"2025-11": ev}, TRAIN_VERSIONS,
            {"2025-11": ["02.09.01"]}, GROUPS, seed=1)
        s = rep["per_month"]["2025-11"]["scores"]["pred_model_a"]
        assert s["n_coverage"] == len(ev) - half
        assert 0.5 <= s["coverage"] <= 1.0


class TestVersionPurity:
    """A month touched by two processors is scored but never pure (honesty)."""

    def test_version_pure_month_is_flagged_pure(self):
        rep = version_shift_report(
            _synthetic_panel(), {"2025-11": _eval_frame()}, TRAIN_VERSIONS,
            {"2025-11": ["02.09.01"]}, GROUPS, seed=1)
        assert rep["eval_version_pure"] is True
        assert rep["mixed_version_months"] == []
        assert rep["per_month"]["2025-11"]["version_pure"] is True

    def test_mixed_version_month_still_scored_but_flagged(self):
        # real manifest case: 2025-12 rows span 02.09.01 and 03.00.DV
        rep = version_shift_report(
            _synthetic_panel(), {"2025-12": _eval_frame(months=("2025-12",))},
            TRAIN_VERSIONS, {"2025-12": ["02.09.01", "03.00.DV"]}, GROUPS,
            seed=1)
        assert rep["eval_version_pure"] is False
        assert rep["mixed_version_months"] == ["2025-12"]
        month = rep["per_month"]["2025-12"]
        assert month["version_pure"] is False
        assert month["versions"] == ["02.09.01", "03.00.DV"]
        assert "pred_model_a" in month["scores"]  # still measured, not dropped


class TestEvalDayPurity:
    """The eval set is chosen on the treatment (processor), never outcomes."""

    def test_measured_2025_11_pattern(self):
        # day-level truth measured from GEE over Delhi's overpass window
        days = {
            "2025-11-03": [],
            "2025-11-20": ["2.8.0"],
            "2025-11-22": ["02.09.01", "2.8.0"],
            "2025-11-23": ["02.09.01"],
            "2025-11-24": ["02.09.01", "03.00.DV"],
        }
        keep, excluded = partition_eval_days(days, ["2.8.0"])
        assert keep == ["2025-11-23"]
        assert excluded["2025-11-03"]["reason"] == "no_overpass_window"
        assert excluded["2025-11-20"]["reason"] == "training_version_present"
        assert excluded["2025-11-22"]["reason"] == "training_version_present"
        assert excluded["2025-11-24"]["reason"] == "mixed_new_versions"

    def test_all_pure_days_are_kept(self):
        days = {f"2025-12-{d:02d}": ["02.09.01"] for d in range(1, 32)}
        keep, excluded = partition_eval_days(days, ["2.8.0"])
        assert len(keep) == 31
        assert excluded == {}


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
