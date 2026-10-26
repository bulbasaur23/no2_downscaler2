"""Version-shift evaluation (Phase 3, the G0 version-drift mitigation).

Concept: TROPOMI processor updates change retrieval distributions; the
Nov-2025 update (02.09.01) is exactly the kind of shift the plan demands we
QUANTIFY, not hand-wave. Protocol enforced here:

1. Everything is fitted on the training window only (2023-04..2025-09,
   transitions excluded): the model's hyperparameters, its final refit on
   all window rows, the climatology/city-mean/Ridge references, and the
   conformal margins (calibrated on window OOF residuals with the same
   season-stratified scheme as in production serving).
2. The evaluation months (from version_shift_start, e.g. 2025-11) are read
   separately, passed through the SAME feature module (train/serve parity),
   and predicted FROZEN - nothing is refit on them, nothing from them may
   flow back into tuning (the plan's never-tune-on-shift-set rule).
3. Reported per method: cellwise RMSE/bias + two-level (level/pattern)
   scores, and the frozen-interval coverage. Coverage on the shift set vs
   the window OOF quantiles is the drift diagnostic: if the processor
   changed the target distribution, miscalibration shows up first.
4. A hard guard: the eval months' processor versions must NOT overlap the
   training-version modal set; asserting the premise makes the test honest.
   A month that mixes two processor versions is scored but flagged
   (`version_pure: false`, `mixed_version_months`) - only version-pure months
   can attribute a change to one retrieval update.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from no2map.baselines import Climatology, DailyCityMean, RidgeBaseline
from no2map.conformal import SplitConformal
from no2map.features import TARGET, build_features, feature_block
from no2map.model import PRED_COLS, XGBConfig, run_blocked_evaluation, season_key
from no2map.evaluation import full_scores


def fit_all(train_panel: pd.DataFrame, groups: list[str], seed: int,
            alpha: float, block_m: float = 30_000.0, buffer_m: float = 0.0,
            xgb: XGBConfig | None = None) -> dict:
    """One blocked-CV pass (OOF + conformal) + final refits on all rows."""
    xgb = xgb or XGBConfig()
    oof, report = run_blocked_evaluation(
        train_panel, groups, block_m=block_m, buffer_m=buffer_m, seed=seed,
        alpha=alpha, xgb=xgb)
    conf = SplitConformal(alpha=alpha, key_fn=season_key).fit(oof)

    rows = build_features(train_panel)
    fe_cols = feature_block(rows, groups)
    model = xgb.make(seed)
    model.fit(rows[fe_cols], rows[TARGET])
    clim = Climatology().fit(rows)
    dcm = DailyCityMean().fit(rows)
    fitted_final: dict = {"pred_model_a": (model, fe_cols),
                          "pred_climatology": clim,
                          "pred_city_mean": dcm}
    try:
        fitted_final["pred_ridge"] = RidgeBaseline().fit(rows, fe_cols, lam=1.0)
    except ValueError:
        fitted_final["pred_ridge"] = None  # degenerate: falls back to climatology

    return {
        "oof_report": report,
        "conformal": conf,
        "final": fitted_final,
    }


def predict_frozen(fitted: dict, eval_df: pd.DataFrame,
                   groups: list[str]) -> pd.DataFrame:
    """Frozen predictions on an eval frame through the shared feature pass."""
    rows = build_features(eval_df)
    out = pd.DataFrame(index=rows.index)
    model, fe_cols = fitted["final"]["pred_model_a"]
    out["pred_model_a"] = model.predict(rows[fe_cols])
    out["pred_climatology"] = fitted["final"]["pred_climatology"].predict(rows).to_numpy()
    out["pred_city_mean"] = fitted["final"]["pred_city_mean"].predict(rows).to_numpy()
    ridge = fitted["final"].get("pred_ridge")
    if ridge is None:
        out["pred_ridge"] = out["pred_climatology"]  # degenerate-fold fallback
    else:
        try:
            out["pred_ridge"] = ridge.predict(rows).to_numpy()
        except ValueError:
            out["pred_ridge"] = out["pred_climatology"]
    out[TARGET] = rows[TARGET].to_numpy()
    out["cell_id"] = rows["cell_id"].to_numpy()
    out["date"] = rows["date"].to_numpy()
    out["month"] = rows["month"].to_numpy()
    return out


def version_shift_report(
    train_panel: pd.DataFrame,
    eval_frames: dict[str, pd.DataFrame],
    train_versions: list[str],
    eval_versions: dict[str, list[str]],
    groups: list[str],
    seed: int = 0,
    alpha: float = 0.1,
    xgb: XGBConfig | None = None,
) -> dict:
    """Full shift report; refuses if the eval versions overlap training's."""
    overlap = sorted(set(train_versions) & set(
        v for vs in eval_versions.values() for v in vs))
    if overlap:
        raise ValueError(
            f"version-shift premise violated: eval months use training "
            f"processor versions {overlap}; the shift set must be new-"
            "processor months (use the g1 report / manifest versions)")

    fitted = fit_all(train_panel, groups, seed, alpha, xgb=xgb)
    mixed = {m: vs for m, vs in eval_versions.items() if len(vs) > 1}
    out: dict = {
        "train_blocked": fitted["oof_report"],
        "training_versions": train_versions,
        "eval_versions": eval_versions,
        # an eval month touched by more than one processor version cannot be
        # attributed to a single retrieval change: score it, but never let it
        # carry the shift claim on its own (reported, not silently pooled)
        "eval_version_pure": not mixed,
        "mixed_version_months": sorted(mixed),
        "per_month": {},
    }
    for month, frame in sorted(eval_frames.items()):
        preds = predict_frozen(fitted, frame, groups)
        conf = fitted["conformal"]
        preds["conf_margin"] = conf.margin(preds)
        cov_row = {
            col: {
                **{k: v for k, v in full_scores(preds, col).items()
                   if k != "conformal"},
                "coverage": float((np.abs(preds[TARGET] - preds[col])
                                   <= preds["conf_margin"]).mean()),
                "mean_width": float(preds["conf_margin"].mean()),
            }
            for col in PRED_COLS if col in preds
        }
        out["per_month"][month] = {
            "versions": eval_versions.get(month, []),
            "version_pure": len(eval_versions.get(month, [])) <= 1,
            "n_rows": int(len(preds)),
            "scores": cov_row,
        }
    return out
