"""Tests for the model artifact (Phase 4 serving contract).

Round-trip is the core claim: fit -> save -> load -> predict must give
IDENTICAL point predictions and margins as the in-memory objects, with no
training code (or pickled state) needed at serve time. Also pinned: the
pooled-quantile fallback for season strata unseen by calibration, and the
hard schema-version pin at load/predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from no2map.artifact import load_model_artifact, save_model_artifact
from no2map.conformal import SplitConformal
from no2map.features import FEATURE_GROUPS, TARGET, build_features, feature_block
from no2map.model import XGBConfig, season_key

from tests.test_model import _synthetic_panel

GROUPS = list(FEATURE_GROUPS)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    df = _synthetic_panel()
    rows = build_features(df)
    fe_cols = feature_block(rows, GROUPS)
    model = XGBConfig(n_estimators=60).make(seed=5)
    model.fit(rows[fe_cols], rows[TARGET])
    resid = rows[TARGET] - model.predict(rows[fe_cols])
    calib = pd.DataFrame({"residual": resid, "month": rows["month"]})
    conf = SplitConformal(alpha=0.1, key_fn=season_key).fit(calib)
    d = tmp_path_factory.mktemp("artifact")
    save_model_artifact(d, model, conf, fe_cols, GROUPS,
                        meta={"city": "delhi_ncr", "months": ["2024-06", "2024-07"]})
    return d, model, conf, rows


class TestRoundTrip:
    def test_predictions_identical_after_reload(self, artifact):
        d, model, _, rows = artifact
        art = load_model_artifact(d)
        out = art.predict(rows)
        np.testing.assert_allclose(
            np.asarray(out["pred"]), model.predict(rows[art.fe_cols]), rtol=0,
            atol=1e-6)

    def test_margins_match_in_memory_conf(self, artifact):
        d, _, conf, rows = artifact
        art = load_model_artifact(d)
        np.testing.assert_allclose(
            np.asarray(art.predict(rows)["margin"]), conf.margin(rows), rtol=0)

    def test_metadata_carried(self, artifact):
        d, _, _, _ = artifact
        art = load_model_artifact(d)
        assert art.meta["city"] == "delhi_ncr"
        assert art.groups == GROUPS
        assert art.alpha == pytest.approx(0.1)


class TestFallbackLadder:
    def test_unseen_season_uses_pooled_quantile(self, artifact):
        d, _, conf, rows = artifact
        art = load_model_artifact(d)
        # training months are 6/7 -> Q1/Q2; probe with month 2 -> Q0 unseen
        probe = rows.copy()
        probe["date"] = "2024-02-01"
        probe["month"] = 2
        out = art.predict(probe)
        assert (np.isclose(out["margin"], conf.q_by_stratum[None])).all()
        # and the pooled root is present in the loaded table
        assert None in art.q_by_stratum


class TestParityPin:
    def test_schema_mismatch_rejected_at_predict(self, artifact, monkeypatch):
        d, _, _, rows = artifact
        # predict reads the installed-vs-artifact version from the artifact
        # module's own binding, so that is what gets pinned
        import no2map.artifact as art_mod
        monkeypatch.setattr(art_mod, "FEATURE_SCHEMA_VERSION", "fm-v999")
        art_obj = load_model_artifact(d)
        with pytest.raises(RuntimeError, match="schema"):
            art_obj.predict(rows)
