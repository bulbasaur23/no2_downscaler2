"""Shared feature module (Phase 2): THE contract between training and live.

Rule from the plan (risk 10, parity): training must NEVER compute a feature
that the live request path cannot produce bit-for-bit. This module is the only
place features are derived from exported/requested signals; both Phase-3
training and the Phase-4 live API call these functions.

Feature groups mirror the Phase-3 ablation axes (weather-only / +lags / +static
/ +temporal) so ablations are a filter, not a search.

FEATURE_SCHEMA_VERSION is pinned: the manifest records which version produced
each Parquet file, and the live path refuses mismatches (parity pin).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FEATURE_SCHEMA_VERSION = "fm-v1-2026-10-24"  # bump on any feature-schema change (parity pin)

TARGET = "value_umol_m2"

# ---- feature groups (ablation axes) --------------------------------------
STATIC_FEATURES = [
    "elevation_mean_m", "elevation_std_m", "elevation_min_m", "elevation_max_m",
    "water_fraction",
]
WEATHER_INST_FEATURES = [
    "t2m_deg_c", "rh_pct", "sp_hpa_inst",
    "wind_speed_inst", "wind_u10_inst", "wind_v10_inst",
    "wind_dir_cos_inst", "wind_dir_sin_inst",
    "blh_m_inst", "ventilation_proxy", "blh_x_precip_proxy",
]
WEATHER_LAG_FEATURES = [
    "wind_speed_l24", "ventilation_l24", "wind_speed_l48",
    "ventilation_l48", "precip_l24_m", "precip_l48_m",
]
TEMPORAL_FEATURES = ["month_sin", "month_cos"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "static": STATIC_FEATURES,
    "weather_inst": WEATHER_INST_FEATURES,
    "weather_lags": WEATHER_LAG_FEATURES,
    "temporal": TEMPORAL_FEATURES,
}


def derivable(df: pd.DataFrame, required: list[str]) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"input dataframe lacks required columns: {missing}")


def _relative_humidity_pct(t2m_k: pd.Series, d2m_k: pd.Series) -> pd.Series:
    """RH from 2 m temperature and dewpoint (Magnus-type inversion; standard
    GFS/ECMWF practice). Pure function of the two live-available signals."""
    def _es(t_c):  # saturation vapour pressure hPa
        return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    t_c, d_c = t2m_k - 273.15, d2m_k - 273.15
    with np.errstate(divide="ignore", invalid="ignore"):
        rh = 100.0 * _es(d_c) / _es(t_c)
    return rh.clip(0, 100)


def _wind_speed(u: pd.Series, v: pd.Series) -> pd.Series:
    return np.hypot(u, v)


def _unit_vec(u: pd.Series, v: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Direction as unit vector components (circular, no wraparound branch cut).
    Zero wind -> (0.5, 0.5): a neutral, defined encoding rather than NaN."""
    speed = _wind_speed(u, v)
    cos = np.where(speed > 1e-6, u / speed, 0.5)
    sin = np.where(speed > 1e-6, v / speed, 0.5)
    return pd.Series(cos, index=u.index), pd.Series(sin, index=u.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all feature columns in place-safe copy. Idempotent.

    Everything here is computable at REQUEST time from data with short
    latency (weather) or static layers - no NO2-derived covariates ever
    enter (that would break the draw-any-polygon live guarantee)."""
    derivable(df, ["wind_u10_inst", "wind_v10_inst", "blh_m_inst",
                   "wind_u10_l24", "wind_v10_l24", "wind_u10_l48", "wind_v10_l48",
                   "t2m_k_inst", "d2m_k_inst", "sp_pa_inst", "month"])
    out = df.copy()

    out["t2m_deg_c"] = out["t2m_k_inst"] - 273.15
    out["rh_pct"] = _relative_humidity_pct(out["t2m_k_inst"], out["d2m_k_inst"])
    out["sp_hpa_inst"] = out["sp_pa_inst"] / 100.0

    out["wind_speed_inst"] = _wind_speed(out["wind_u10_inst"], out["wind_v10_inst"])
    cos_d, sin_d = _unit_vec(out["wind_u10_inst"], out["wind_v10_inst"])
    out["wind_dir_cos_inst"] = cos_d
    out["wind_dir_sin_inst"] = sin_d

    out["ventilation_proxy"] = out["wind_speed_inst"] * out["blh_m_inst"] / 1000.0
    out["blh_x_precip_proxy"] = out["blh_m_inst"] / 1000.0 * out["precip_l24_m"].fillna(0)

    out["wind_speed_l24"] = _wind_speed(out["wind_u10_l24"], out["wind_v10_l24"])
    out["wind_speed_l48"] = _wind_speed(out["wind_u10_l48"], out["wind_v10_l48"])
    out["ventilation_l24"] = out["wind_speed_l24"] * out["blh_m_inst"] / 1000.0
    out["ventilation_l48"] = out["wind_speed_l48"] * out["blh_m_inst"] / 1000.0

    m = out["month"].astype(int)
    out["month_sin"] = np.sin(2 * math.pi * (m - 1) / 12.0)
    out["month_cos"] = np.cos(2 * math.pi * (m - 1) / 12.0)

    return out


def feature_block(df: pd.DataFrame, groups: list[str]) -> list[str]:
    """Ordered feature columns for a set of ablation groups (G3 axes)."""
    cols: list[str] = []
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise KeyError(f"unknown feature group {g!r}; have {sorted(FEATURE_GROUPS)}")
        cols.extend(FEATURE_GROUPS[g])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"feature columns not built yet (run build_features): {missing}")
    return cols  # ordered, unique (groups don't overlap)


def clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for training: OBSERVED state with a target and complete features.

    Parity note: at request time the same filter marks cells that can be
    scored rather than guessed; training never sees rows the live path
    could not produce."""
    ok = (df["state"] == "observed") & df[TARGET].notna()
    return df.loc[ok].reset_index(drop=True)
