"""Two-level scoring + yardstick metrics (Phase 3, plan section 6).

The plan's honest-decomposition rule: report skill at TWO levels, because a
model can look brilliant while only learning the city mean:

(a) LEVEL: does the model track the seasonal/weather-driven day-to-day
    variation of the city's mean? Scored on per-DATE city means (predicted
    vs observed), using RMSE and Pearson correlation across dates.
(b) PATTERN: given the day's level, does the model recover the *spatial*
    structure within the day? Scored as the mean per-date correlation
    between per-cell predictions and observations, with a same-date-only
    pairing guard (min_cells per date) - never pooled across dates, which
    would re-admit the level signal and inflate the pattern score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from no2map.features import TARGET


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def bias(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.asarray(p, float) - np.asarray(y, float)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    """Coefficient of determination (plan section 6 metric list)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")  # constant target: R2 undefined, not 1.0
    return 1.0 - ss_res / ss_tot


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        raise ValueError("pearson needs >= 3 points")
    xs = x.std()
    ys = y.std()
    if xs < 1e-12 or ys < 1e-12:
        return float("nan")  # degenerate: constant input, correlation undefined
    return float(((x - x.mean()) * (y - y.mean())).mean() / (xs * ys))


def level_scores(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    """City-mean tracking: per-date mean(pred) vs mean(target).

    Uses only dates that have BOTH observed and predicted cells; a date with
    no observed cells carries no level information and must stay out."""
    g = df.groupby("date")[TARGET]
    obs_daily = g.mean()
    pred_daily = df.groupby("date")[pred_col].mean()
    shared = obs_daily.index.intersection(pred_daily.index)
    return {
        "n_dates": int(len(shared)),
        "rmse": rmse(obs_daily.loc[shared].to_numpy(), pred_daily.loc[shared].to_numpy()),
        "r": pearson(obs_daily.loc[shared].to_numpy(), pred_daily.loc[shared].to_numpy()),
    }


def pattern_scores(df: pd.DataFrame, pred_col: str = "pred",
                   min_cells: int = 10) -> dict:
    """Within-day spatial structure: mean per-date cellwise correlation."""
    r_by_date = []
    for _, day in df.groupby("date"):
        y = day[TARGET].to_numpy(dtype=float)
        p = day[pred_col].to_numpy(dtype=float)
        m = np.isfinite(y) & np.isfinite(p)
        if m.sum() >= min_cells:
            r_by_date.append(pearson(y[m], p[m]))
    valid = np.asarray(r_by_date, dtype=float)
    valid = valid[np.isfinite(valid)]
    return {
        "n_dates": len(r_by_date),
        "n_valid_dates": int(len(valid)),
        "mean_r": float(valid.mean()) if len(valid) else float("nan"),
    }


def anomaly_scores(df: pd.DataFrame, pred_col: str = "pred",
                   ref_col: str = "pred_climatology") -> dict:
    """Skill on anomalies, relative to the climatology anchor (G3 fallback).

    The anchor is the OUT-OF-FOLD climatology prediction (fitted on training
    rows of each fold only), so nothing about the held-out rows enters it.
    Because the same anchor is subtracted from every prediction, anomaly RMSE
    is identical to plain RMSE - what carries extra information is:
      - r_anomaly: does the method track departures from climatology
        (temporal and spatial) rather than only reproducing the mean state?
      - skill_vs_climatology: 1 - MSE(method)/MSE(climatology), the standard
        skill score (positive = better than climatology).
    Climatology scored against itself is (nan, 0.0) by construction.
    """
    if ref_col not in df.columns:
        return {"r_anomaly": float("nan"), "skill_vs_climatology": float("nan")}
    ok = (np.isfinite(df[TARGET].to_numpy(dtype=float)) &
          np.isfinite(df[pred_col].to_numpy(dtype=float)) &
          np.isfinite(df[ref_col].to_numpy(dtype=float)))
    d = df.loc[ok]
    if d.empty:
        return {"r_anomaly": float("nan"), "skill_vs_climatology": float("nan")}
    obs_anom = d[TARGET].to_numpy(dtype=float) - d[ref_col].to_numpy(dtype=float)
    pred_anom = d[pred_col].to_numpy(dtype=float) - d[ref_col].to_numpy(dtype=float)
    mse_ref = float(np.mean(obs_anom ** 2))
    if mse_ref < 1e-12:
        skill = float("nan")
    else:
        skill = 1.0 - float(np.mean((obs_anom - pred_anom) ** 2)) / mse_ref
    r = float("nan")
    if pred_col != ref_col and len(obs_anom) >= 3:
        try:
            r = pearson(obs_anom, pred_anom)
        except ValueError:
            r = float("nan")
    return {"r_anomaly": r,
            "skill_vs_climatology": 0.0 if pred_col == ref_col else skill}


def full_scores(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    """Cellwise RMSE/MAE/R2/bias + both two-level scores - the Phase-3
    runner's single scoring entrypoint so the baselines and the model are
    scored through the exact same code path (always-shown rule)."""
    m = np.isfinite(df[TARGET].to_numpy(dtype=float)) & np.isfinite(df[pred_col].to_numpy(dtype=float))
    d = df.loc[m]
    out = {
        "n": int(m.sum()),
        "rmse": rmse(d[TARGET].to_numpy(), d[pred_col].to_numpy()),
        "mae": mae(d[TARGET].to_numpy(), d[pred_col].to_numpy()),
        "bias": bias(d[TARGET].to_numpy(), d[pred_col].to_numpy()),
        "r2": r2(d[TARGET].to_numpy(), d[pred_col].to_numpy()),
    }
    out.update({f"level_{k}": v for k, v in level_scores(d, pred_col).items()})
    out.update({f"pattern_{k}": v for k, v in pattern_scores(d, pred_col).items()})
    return out
