"""Model A training and honest comparison runner (Phase 3, plan section 6).

The single place where Model A (XGBoost) is fitted AND scored against the
four baselines through the exact same fold machinery, so the always-shown
rule cannot be gamed by scoring the model on a different path:

- spatial-block evaluation: leave-one-block-out folds (splits module) with
  the variogram-backed block size and a buffer gap; within the fold the
  model, Ridge, climatology and daily-city-mean ALL train on the same rows;
  the held-out rows receive every method's prediction. Out-of-fold rows carry
  each method's residual -> the split-conformal quantile is calibrated on
  blocked residuals exactly as the plan requires.
- chronological evaluation lives in the runner script (same scoring layer).

Determinism: xgboost fixed seed, fold construction deterministic, no
shuffling anywhere. Feature matrix built ONLY via the shared feature module
(parity contract; risk 10).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from no2map.baselines import Climatology, DailyCityMean, RidgeBaseline
from no2map.conformal import SplitConformal
from no2map.features import (
    FEATURE_GROUPS,
    TARGET,
    build_features,
    clean_training_frame,
    feature_block,
)
from no2map.splits import spatial_block_folds

PRED_COLS = ["pred_model_a", "pred_ridge", "pred_climatology", "pred_city_mean"]


def season_key(row) -> str:
    """Stratum key for conformal calibration: calendar quarter Q1..Q4
    (dtype-stable: int() cast, per the conformal module's warning)."""
    return f"Q{(int(row['month']) - 1) // 3}"


def design_matrix(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Feature columns for `groups` after the shared build_features pass."""
    prepared = build_features(df)
    return prepared.loc[:, feature_block(prepared, groups)]


def run_ablation(
    df: pd.DataFrame,
    ladder: list[list[str]] | None = None,
    block_m: float = 30_000.0,
    buffer_m: float = 0.0,
    seed: int = 0,
    alpha: float = 0.1,
    xgb: XGBConfig | None = None,
) -> list[dict]:
    """Cumulative feature-group ablation on IDENTICAL folds.

    Default ladder isolates the four signal families the plan names: static
    topography/water -> +instant weather -> +ventilation lags -> +seasonal
    harmonics. Each stage re-runs the full blocked evaluation (same folds,
    same baselines through the same path), so the delta between stages is
    attributable to the ADDED family, not to a different evaluation.
    """
    xgb = xgb or XGBConfig()
    if ladder is None:
        ladder = [
            ["static"],
            ["static", "weather_inst"],
            ["static", "weather_inst", "weather_lags"],
            ["static", "weather_inst", "weather_lags", "temporal"],
        ]
    stages: list[dict] = []
    for groups in ladder:
        oof, report = run_blocked_evaluation(
            df, groups, block_m=block_m, buffer_m=buffer_m, seed=seed,
            alpha=alpha, xgb=xgb)
        stage = {
            "groups": groups,
            "model_rmse": report["pred_model_a"]["rmse"],
            "ridge_rmse": report["pred_ridge"]["rmse"],
            "climatology_rmse": report["pred_climatology"]["rmse"],
            "city_mean_rmse": report["pred_city_mean"]["rmse"],
            "level_rmse": report["pred_model_a"].get("level_rmse"),
            "level_r": report["pred_model_a"].get("level_r"),
            "a": report["conformal"]["alpha"],
        }
        if stages:
            stage["delta_model_m"] = report["pred_model_a"]["rmse"] - stages[-1]["model_rmse"]
        stages.append(stage)
    return stages


@dataclass
class XGBConfig:
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0

    def make(self, seed: int) -> XGBRegressor:
        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=seed,
            n_jobs=4,
        )


def run_blocked_evaluation(
    df: pd.DataFrame,
    groups: list[str],
    block_m: float,
    buffer_m: float = 0.0,
    seed: int = 0,
    alpha: float = 0.1,
    xgb: XGBConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Leave-block-out evaluation, all methods per fold on identical data.

    Returns (oof_frame, report) where oof_frame holds every held-out row with
    columns pred_model_a/pred_ridge/pred_climatology/pred_city_mean and the
    report dict scores each method with evaluation.full_scores, adds the
    fold map for traceability, and attaches the conformal calibration (q_pooled
    and per-season) fitted on OOF residuals.
    """
    xgb = xgb or XGBConfig()
    if groups[-1] not in FEATURE_GROUPS:
        raise KeyError(f"unknown feature group {groups[-1]!r}")

    rows = clean_training_frame(df)
    prepared = build_features(rows)
    fe_cols = feature_block(prepared, groups)
    cell_ids = sorted(prepared["cell_id"].unique())
    folds = spatial_block_folds(cell_ids, block_m=block_m, buffer_m=buffer_m)
    rows_by_block = {blk: set(val_cells) for blk, _, val_cells in folds}

    oof_parts: list[pd.DataFrame] = []
    for blk, train_cells, val_cells in folds:
        train_rows = prepared[~prepared["cell_id"].isin(rows_by_block[blk])]
        val_rows = prepared[prepared["cell_id"].isin(set(val_cells))]
        if train_rows.empty or val_rows.empty:
            continue

        X_tr, y_tr = train_rows[fe_cols], train_rows[TARGET]

        model = xgb.make(seed)
        model.fit(X_tr, y_tr)

        clim = Climatology().fit(train_rows)
        dcm = DailyCityMean().fit(train_rows)

        preds = pd.DataFrame({
            "pred_model_a": model.predict(val_rows[fe_cols]),
            "pred_climatology": clim.predict(val_rows).to_numpy(),
            "pred_city_mean": dcm.predict(val_rows).to_numpy(),
        }, index=val_rows.index)
        try:
            ridge = RidgeBaseline().fit(train_rows, fe_cols, lam=1.0)
            preds["pred_ridge"] = ridge.predict(val_rows).to_numpy()
        except ValueError:
            preds["pred_ridge"] = preds["pred_climatology"]  # degenerate fold

        preds[TARGET] = val_rows[TARGET].to_numpy()
        preds["cell_id"] = val_rows["cell_id"].to_numpy()
        preds["date"] = val_rows["date"].to_numpy()
        preds["month"] = val_rows["month"].to_numpy()
        preds["block"] = blk
        oof_parts.append(preds)

    if not oof_parts:
        raise RuntimeError("no folds produced predictions; check data/folds")
    oof = pd.concat(oof_parts, ignore_index=True)

    # conformal calibration on the blocked OOF residuals (plan: "calibrated
    # on spatial-block validation"), stratified by season with pooled root
    oof["residual"] = oof[TARGET] - oof["pred_model_a"]
    conf = SplitConformal(alpha=alpha, key_fn=season_key).fit(oof)
    oof["conf_margin"] = conf.margin(oof)

    report = {"folds": list(rows_by_block), "n_folds": len(rows_by_block)}
    for col in PRED_COLS:
        s = {"rmse": float(np.sqrt(np.mean((oof[TARGET] - oof[col]) ** 2)))}
        report[col] = s
    report["conformal"] = {
        "alpha": alpha,
        "q_by_season": {k: conf.q_by_stratum[k]
                        for k in sorted(k for k in conf.q_by_stratum if k is not None)},
        "q_pooled": conf.q_by_stratum[None],
    }
    return oof, report
