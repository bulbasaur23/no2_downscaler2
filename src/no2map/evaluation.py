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
    if not r_by_date:
        return {"n_dates": 0, "mean_r": float("nan")}
    return {
        "n_dates": len(r_by_date),
        "mean_r": float(np.nanmean(r_by_date)),
    }


def full_scores(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    """Cellwise RMSE/MAE/bias + both two-level scores - the Phase-3 runner's
    single scoring entrypoint so the baselines and the model are scored
    through the exact same code path (always-shown rule)."""
    m = np.isfinite(df[TARGET].to_numpy(dtype=float)) & np.isfinite(df[pred_col].to_numpy(dtype=float))
    d = df.loc[m]
    out = {
        "n": int(m.sum()),
        "rmse": rmse(d[TARGET].to_numpy(), d[pred_col].to_numpy()),
        "mae": mae(d[TARGET].to_numpy(), d[pred_col].to_numpy()),
        "bias": bias(d[TARGET].to_numpy(), d[pred_col].to_numpy()),
    }
    out.update({f"level_{k}": v for k, v in level_scores(d, pred_col).items()})
    out.update({f"pattern_{k}": v for k, v in pattern_scores(d, pred_col).items()})
    return out
