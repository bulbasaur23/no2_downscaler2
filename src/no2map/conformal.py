"""Split-conformal intervals (Phase 3, plan section 6).

Concept: an XGBoost point prediction says nothing about uncertainty. Split-
conformal gives DISTRIBUTION-FREE intervals: calibration residuals |y - p|
from an independent validation set are sorted; the q-th order statistic (q =
ceil((n+1)(1-alpha))/n, the conformal-corrected quantile) gives a symmetric
margin whose coverage is >= 1-alpha on exchangeable held-out data, with NO
distributional assumption and no access to model internals.

Plan requirements honored here:
- calibration happens on the spatial-BLOCKED validation fold (the caller
  picks the residuals; the splits module provides the fold);
- stratified calibration (by season and by distance-to-nearest-valid-cell)
  via `key_fn`, with a pooled fallback when a stratum is unseen at predict
  time - the same explicit-fallback pattern as the climatology baseline;
- coverage/width are reported on blocked, chronological, version-shift and
  held-out-city sets by the Phase-3 runner.

DTYPE WARNING (caught by tests): `df.apply(axis=1)` upcasts int columns to
float on mixed-dtype frames, so a key_fn using str(row["month"]) maps "1"
and "1.0" to DIFFERENT strata depending on which frame it sees first. Key
functions must be dtype-stable: use int(...) or a dict keyed by the raw
value, never str() of a numeric column.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """1-alpha upper bound for |residual| on exchangeable new data."""
    n = len(residuals)
    if n == 0:
        raise ValueError("no calibration residuals")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        # not enough calibration points for the requested level: use the max
        return float(np.max(np.abs(residuals)))
    return float(np.sort(np.abs(residuals))[k - 1])


@dataclass
class SplitConformal:
    """Calibrated symmetric interval widths; `key_fn` rows -> stratum or None."""

    alpha: float
    key_fn: "callable | None" = None
    q_by_stratum: dict | None = None

    def fit(self, calib: pd.DataFrame, residual_col: str = "residual") -> "SplitConformal":
        if self.key_fn is None:
            keys = pd.Series([None] * len(calib), index=calib.index)
        else:
            if len(calib) == 0:
                raise ValueError("calibration frame is empty")
            keys = calib.apply(self.key_fn, axis=1)
        self.q_by_stratum = {
            k: conformal_quantile(g[residual_col].to_numpy(dtype=float), self.alpha)
            for k, g in calib.groupby(keys)
        }
        # fallback root: pooled quantile for strata unseen at predict time
        self.q_by_stratum[None] = conformal_quantile(
            calib[residual_col].to_numpy(dtype=float), self.alpha)
        return self

    def margin(self, df: pd.DataFrame) -> np.ndarray:
        """Symmetric half-width [p - m, p + m] per row; unfitted -> raise."""
        if self.q_by_stratum is None:
            raise RuntimeError("SplitConformal not calibrated (call fit)")
        m = np.empty(len(df), dtype=float)
        for i in range(len(df)):
            if self.key_fn is None:
                m[i] = self.q_by_stratum[None]
            else:
                k = self.key_fn(df.iloc[i])
                m[i] = self.q_by_stratum.get(k, self.q_by_stratum[None])
        return m

    def coverage(self, df: pd.DataFrame, point_col: str, truth_col: str) -> float:
        """Empirical covered fraction on any evaluation frame (the plan's
        reported coverage; width = mean of the margins)."""
        resid = (df[truth_col] - df[point_col]).to_numpy(dtype=float)
        return float((np.abs(resid) <= self.margin(df)).mean())
