"""Phase-5 transfer + OOD tests (gate G5).

Synthetic cities stand in for Bengaluru/Kolkata: the train/test split, the
frozen scoring, and the OOD gate are exercised without Earth Engine. The OOD
flag's load-bearing property is tested directly: flagged rows must carry
higher held-out error, and an empty group must yield `None`, never a yes.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from no2map.features import FEATURE_GROUPS, TARGET
from no2map.model import PRED_COLS
from no2map.transfer import (
    OOD_FEATURES,
    OODReference,
    city_diagnostics,
    ood_evidence,
    transfer_report,
)

from tests.test_model import _synthetic_panel

GROUPS = list(FEATURE_GROUPS)


def _city(seed: int = 3) -> pd.DataFrame:
    return _synthetic_panel(seed=seed)


def _shifted(df: pd.DataFrame, delta: float,
             n_cells: int | None = None) -> pd.DataFrame:
    """Raise elevation for all (or the first n) cells - an OOD stand-in city."""
    out = df.copy()
    cells = sorted(out["cell_id"].unique())
    if n_cells is not None:
        cells = cells[:n_cells]
    mask = out["cell_id"].isin(cells)
    for col in ("elevation_mean_m", "elevation_min_m", "elevation_max_m"):
        out.loc[mask, col] = out.loc[mask, col] + delta
    return out


class TestOODReference:
    def test_shifted_city_flagged_in_distribution_not(self):
        ref = OODReference.fit(_city(), quantile=0.95)
        same = ref.distances(_city(seed=4))
        shifted = ref.distances(_shifted(_city(seed=4), 400.0))
        # a 95% gate may flag up to ~5% of in-distribution cells by design
        assert (same["ood_distance"] > ref.threshold).mean() <= 0.1
        assert (shifted["ood_distance"] > ref.threshold).all()

    def test_missing_statics_count_as_out_of_distribution(self):
        # unknown context must never be assumed in-distribution
        ref = OODReference.fit(_city())
        df = _city(seed=4)
        victim = df["cell_id"].iloc[0]
        df.loc[df["cell_id"] == victim, "elevation_mean_m"] = np.nan
        d = ref.distances(df).set_index("cell_id")
        assert np.isinf(d.loc[victim, "ood_distance"])

    def test_reference_serializes_without_raw_points(self):
        ref = OODReference.fit(_city())
        d = ref.to_dict()
        assert d["features"] == OOD_FEATURES
        assert d["n_reference_cells"] > 0
        assert "z" not in d
        json.dumps(d)  # the report is JSON, so this must not raise

    def test_loo_threshold_is_positive_and_below_shifted(self):
        ref = OODReference.fit(_city())
        assert ref.threshold > 0.0
        assert ref.threshold < 400.0


class TestOODEvidence:
    def _preds(self) -> pd.DataFrame:
        return pd.DataFrame({
            "cell_id": ["a", "b", "c", "d"],
            TARGET: [1.0, 1.0, 1.0, 1.0],
            "pred_model_a": [1.1, 0.9, 4.0, 4.0],
            "ood_distance": [0.1, 0.2, 5.0, 6.0],
            "ood": [False, False, True, True],
        })

    def test_flagged_group_higher_error_is_reported(self):
        ev = ood_evidence(self._preds())
        assert ev["flagged_higher_error"] is True
        assert ev["rmse_flagged"] > ev["rmse_unflagged"]
        assert ev["n_flagged_cells"] == 2
        assert ev["delta_flagged_minus_unflagged"] > 0

    def test_empty_group_is_not_evidence(self):
        preds = self._preds()
        preds["ood"] = False
        ev = ood_evidence(preds)
        assert ev["rmse_flagged"] is None
        assert ev["flagged_higher_error"] is None

    def test_rows_without_observation_are_excluded(self):
        preds = self._preds()
        preds.loc[4] = {"cell_id": "e", TARGET: np.nan, "pred_model_a": 999.0,
                        "ood_distance": 9.0, "ood": True}
        ev = ood_evidence(preds)
        assert ev["n_scored"] == 4  # the NaN-target row is not scored

    def test_missing_columns_are_rejected(self):
        with pytest.raises(KeyError, match="OOD evidence needs"):
            ood_evidence(pd.DataFrame({TARGET: [1.0]}))


class TestCityDiagnostics:
    def test_state_counts_and_observed_fraction(self):
        df = _city()
        df.loc[df.index[:10], "state"] = "insufficient"
        df.loc[df.index[:10], TARGET] = np.nan
        d = city_diagnostics(df)
        assert d["n_rows"] == len(df)
        assert d["n_cells"] == df["cell_id"].nunique()
        assert d["state_counts"]["insufficient"] == 10
        assert d["observed_fraction"] == pytest.approx((len(df) - 10) / len(df))
        assert d["target_median"] == pytest.approx(df[TARGET].median())


class TestTransferReport:
    def test_fold_is_frozen_and_scores_every_method(self):
        panels = {"alpha": _city(seed=3),
                  "beta": _shifted(_city(seed=4), 400.0, n_cells=10)}
        rep = transfer_report(panels, GROUPS, test_cities=["beta"], seed=1)
        fold = rep["folds"]["beta"]
        assert fold["train_cities"] == ["alpha"]
        assert set(PRED_COLS) <= set(fold["scores"])
        s = fold["scores"]["pred_model_a"]
        assert s["n_coverage"] == len(panels["beta"])  # every synthetic row observed
        assert 0.0 <= s["coverage"] <= 1.0
        # the flagged (displaced) cells really do carry larger error here
        assert fold["ood"]["flagged_higher_error"] is True
        assert rep["cities"]["beta"]["n_cells"] == \
            panels["beta"]["cell_id"].nunique()
        assert rep["protocol"]["ood"]["quantile"] == pytest.approx(0.95)

    def test_unknown_test_city_rejected(self):
        with pytest.raises(KeyError, match="not in panels"):
            transfer_report({"alpha": _city()}, GROUPS, test_cities=["beta"])

    def test_single_city_cannot_train_itself(self):
        with pytest.raises(ValueError, match="no other city"):
            transfer_report({"alpha": _city()}, GROUPS, test_cities=["alpha"])
