"""Phase 2 baselines: the honesty anchors every model must beat.

Plan section 6 requires four baselines, always visible in the final UI:

1. **Per-cell monthly climatology** - median per (cell, month) computed from
   TRAINING years only. Captures "what this place is usually like this time
   of year". The G3 check: Model A must beat THIS by more than seed noise,
   or the deep model is not adding value.
2. **Daily city mean** - the spatial mean of observed cells per day, then
   repeated for every cell. Captures the day's level but nothing spatial.
   Exposes level-vs-pattern skill explicitly (two-level scoring).
3. **Static-only model** - features that never change (terrain, water). If
   this is nearly as good as the full model, weather adds nothing and the
   added complexity is unjustified.
4. **Ridge** on the full feature set - the linear yardstick for XGBoost.

Integrity rules (risk: optimistic evaluation):
- Fitters accept ONLY training-window rows. Callers pass `df_train` already
  restricted; helpers here never infer a window.
- Every baseline handles the LIVE case (any cell, any date, including dates/
  cells never seen in training) through an explicit fallback ladder, so the
  Phase-4 polygon path can always emit a prediction or an honest nodata.

All models here are dependency-free (numpy closed forms) so Phase 2 requires
no new packages and the math is auditable line by line.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from no2map.features import TARGET

# columns every baseline expects on its input frames
KEY_COLS = ("cell_id", "date", "month")


def _check_columns(df: pd.DataFrame, additional: tuple[str, ...] = ()) -> None:
    missing = [c for c in (*KEY_COLS, *additional) if c not in df.columns]
    if missing:
        raise KeyError(f"input frame lacks required columns: {missing}")


# ---------------------------------------------------------------------------
# 1. Per-cell monthly climatology
# ---------------------------------------------------------------------------
@dataclass
class Climatology:
    """Median observed value per (cell_id, month), fitted on training rows.

    Fallback ladder for unseen combinations (all from the training fit, in
    coarse-to-fine order at predict time):
        (cell, month) -> cell median -> month city median -> city median
    """

    by_cell_month: pd.DataFrame | None = None
    by_cell: pd.Series | None = None
    by_month: pd.Series | None = None
    city_median: float = float("nan")

    def fit(self, df: pd.DataFrame) -> "Climatology":
        _check_columns(df, ("month",))
        g = df.groupby(["cell_id", "month"], sort=False)[TARGET]
        table = g.median().rename("value").reset_index()
        self.by_cell_month = table
        self.by_cell = df.groupby("cell_id")[TARGET].median()
        self.by_month = df.groupby("month")[TARGET].median()
        self.city_median = float(df[TARGET].median())
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.by_cell_month is None:
            raise RuntimeError("climatology not fitted")
        lk = df[["cell_id", "month"]].merge(
            self.by_cell_month, on=["cell_id", "month"], how="left"
        )
        preds = lk["value"].to_numpy(dtype=float)

        cell_stats = df["cell_id"].map(self.by_cell).to_numpy(dtype=float)
        month_stats = df["month"].map(self.by_month).to_numpy(dtype=float)

        preds = np.where(np.isnan(preds), cell_stats, preds)
        preds = np.where(np.isnan(preds), month_stats, preds)
        preds = np.where(np.isnan(preds), self.city_median, preds)
        return pd.Series(preds, index=df.index, name="pred_climatology")

    def fallback_rate(self, df: pd.DataFrame) -> float:
        """Fraction of rows whose (cell, month) is unseen - a coverage report,
        not a prediction. High rates mean the evaluate window or region leaks
        outside what training could characterize."""
        if self.by_cell_month is None:
            raise RuntimeError("climatology not fitted")
        seen = self.by_cell_month.set_index(["cell_id", "month"]).index
        probe = pd.MultiIndex.from_frame(df[["cell_id", "month"]])
        return float((~probe.isin(seen)).mean())


# ---------------------------------------------------------------------------
# 2. Daily city mean
# ---------------------------------------------------------------------------
@dataclass
class DailyCityMean:
    """Mean of all OBSERVED cells per day, broadcast to every cell.

    Only spatial-level information: every cell gets the same value on a day.
    Fallback ladder at predict: date mean -> month-of-year mean -> city mean.
    Assumes `date` is a string 'YYYY-MM-DD' column (as in the exports).
    """

    by_date: pd.Series | None = None
    by_month_of_year: pd.Series | None = None
    city_mean: float = float("nan")

    @staticmethod
    def _month_of_year(date_str: pd.Series) -> pd.Series:
        # 'YYYY-MM-DD' -> 'MM'; string slicing keeps it dtype-stable
        return date_str.str.slice(5, 7)

    def fit(self, df: pd.DataFrame) -> "DailyCityMean":
        _check_columns(df, ("date",))
        self.by_date = df.groupby("date")[TARGET].mean()
        # mean of the per-date means: days with more observed cells (clear
        # winter days) should not overweight level estimation. Group by the
        # month extracted from the DATE INDEX (alignment, not row position).
        moy_index = pd.Index([d[5:7] for d in self.by_date.index])
        self.by_month_of_year = pd.Series(
            self.by_date.to_numpy(), index=moy_index
        ).groupby(level=0).mean()
        self.city_mean = float(self.by_date.mean())
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.by_date is None:
            raise RuntimeError("daily city mean not fitted")
        day = df["date"].map(self.by_date).to_numpy(dtype=float)
        moy = self._month_of_year(df["date"]).map(self.by_month_of_year).to_numpy(dtype=float)
        out = np.where(np.isnan(day), moy, day)
        out = np.where(np.isnan(out), self.city_mean, out)
        return pd.Series(out, index=df.index, name="pred_city_mean")


# ---------------------------------------------------------------------------
# 3/4. Ridge (covers static-only AND full-feature Ridge: statutory groups)
# ---------------------------------------------------------------------------
@dataclass
class RidgeBaseline:
    """Closed-form ridge on standardized features; intercept unpenalized.

    Standardization makes the penalty scale-invariant (policy-mass and
    elevation-range changes must not change the model). Features are
    median-imputed with TRAIN medians so eval NaNs cannot leak statistics.
    """

    feature_cols: list[str] = field(default_factory=list)
    coef: np.ndarray | None = None
    intercept: float = float("nan")
    means: np.ndarray | None = None
    stds: np.ndarray | None = None
    medians: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, feature_cols: list[str], lam: float = 1.0) -> "RidgeBaseline":
        if not feature_cols:
            raise ValueError("feature_cols must be non-empty (use a feature group)")
        _check_columns(df)
        X_raw = df[feature_cols].to_numpy(dtype=float)
        y = df[TARGET].to_numpy(dtype=float)

        self.feature_cols = list(feature_cols)
        self.medians = np.nanmedian(np.where(np.isnan(X_raw), np.nan, X_raw), axis=0)
        # constant (all-NaN, degenerate) columns inherit 0 medians; stds pin 0
        self.medians = np.where(np.isnan(self.medians), 0.0, self.medians)
        X = np.where(np.isnan(X_raw), self.medians, X_raw)

        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0)
        self.stds = np.where(self.stds < 1e-12, 1.0, self.stds)
        Z = (X - self.means) / self.stds

        Z1 = np.column_stack([np.ones(len(Z)), Z])
        penalty = np.eye(Z1.shape[1]) * lam
        penalty[0, 0] = 0.0  # intercept unpenalized
        w = np.linalg.solve(Z1.T @ Z1 + penalty, Z1.T @ y)

        self.intercept = float(w[0])
        self.coef = w[1:]
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.coef is None:
            raise RuntimeError("ridge not fitted")
        _check_columns(df)
        X_raw = df[self.feature_cols].to_numpy(dtype=float)
        X = np.where(np.isnan(X_raw), self.medians, X_raw)
        Z = (X - self.means) / self.stds
        return pd.Series(self.intercept + Z @ self.coef, index=df.index,
                         name="pred_ridge")

    def is_static_only(self) -> bool:
        """True when fitted exclusively on the static feature group (the
        'static-only model' baseline of the plan)."""
        from no2map.features import STATIC_FEATURES  # local: avoids cycles
        return self.feature_cols == STATIC_FEATURES
