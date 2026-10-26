"""Residual semivariogram -> spatial block size (Phase 2, Gate G2).

Concept: neighbouring cells' NO2 residuals are correlated (shared weather,
plumes, one orbit's error). Random k-fold CV then leaks: a validation sample
with near-duplicate training samples measures interpolation skill, not spatial
generalization. The honest protocol (plan section 6) blocks the map into
regions LARGER than the residual autocorrelation range with a buffer gap.

This module estimates that range:
1. empirical semivariogram gamma(d): mean of squared differences of
   SAME-DATE residual pairs, binned by pairwise centroid distance (same-date
   is essential - cross-date pairs mix day-level noise into the spatial
   structure, inflating gamma everywhere),
2. exponential-model fit -> the correlation-range parameter r (atmospheric
   residuals rise steeply and trail off; the exponential shape matches), with
   the plan's threshold rule as fallback,
3. block size = r x buffer (plan: "blocks larger than the autocorrelation
   range with a buffer gap"), and
4. the Gate G2 rule: if a single block would exceed ~half of Delhi's extent,
   spatial blocking is impractical - use chronological validation plus
   coarse blocks, and rest the spatial claim on leave-one-city-out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from no2map.grid import parse_cell_id, cell_from_row_col

# IUGG mean radius, same sphere as the grid module's area math
_EARTH_RADIUS_M = 6371008.8

TROPOMI_CELL_M = 0.05 * _EARTH_RADIUS_M * np.pi / 180.0  # ~5557 m north-south


def cell_centroids(cell_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Cell-id strings -> (lat, lon) centroid arrays, ORDER-PRESERVING, via the
    ONE grid module (re-deriving centroids here would risk a second definition
    of 'cell'). Callers align values with these centroids positionally, so the
    function must never re-sort: a silent re-order would pair each value with
    a neighbour's coordinates and (as caught by test) flatten the variogram."""
    lats = np.empty(len(cell_ids))
    lons = np.empty(len(cell_ids))
    for i, cid in enumerate(cell_ids):
        cell = cell_from_row_col(*parse_cell_id(cid))
        lats[i] = cell.center_lat
        lons[i] = cell.center_lon
    return lats, lons


def _haversine_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distance matrix (metres)."""
    p = np.radians(lat)
    lam = np.radians(lon)
    dp = p[:, None] - p[None, :]
    dl = lam[:, None] - lam[None, :]
    a = np.sin(dp / 2) ** 2 + np.cos(p)[:, None] * np.cos(p)[None, :] * np.sin(dl / 2) ** 2
    return 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass
class SemivariogramResult:
    bin_centers_m: np.ndarray
    gamma: np.ndarray          # empirical semivariance per bin
    n_pairs: np.ndarray        # support per bin (honest reporting: thin tails)
    sill: float
    range_m: float             # exponential-fit correlation range (threshold
                               # rule fallback may report the crossing lag)
    fit_rmse: float
    method: str                # 'exponential' | 'threshold' (fallback)
    n_dates_used: int


def empirical_semivariogram(
    df: pd.DataFrame,
    value_col: str,
    n_bins: int = 16,
    max_dist_m: float | None = None,
    max_pairs_per_date: int = 100_000,
) -> SemivariogramResult:
    """Pooled same-date semivariogram of `value_col` (e.g. model residuals).

    Pair budget: per date, at most `max_pairs_per_date` pairs by using the
    first k=ceil(sqrt(2*max_pairs)) cells in deterministic cell-id order.
    Pairs counted once (i < j), plus only strictly-finite values.
    """
    if "date" not in df.columns or "cell_id" not in df.columns:
        raise KeyError("df needs 'date' and 'cell_id' columns")

    # default survey window: 30 cell widths (~167 km) covers the NCR with
    # margin; ranges beyond that are irrelevant for blocking policy
    if max_dist_m is None:
        max_dist_m = 30.0 * TROPOMI_CELL_M

    k_cells = max(2, int(np.ceil(np.sqrt(2.0 * max_pairs_per_date))))
    gam_sum = np.zeros(n_bins)
    pair_cnt = np.zeros(n_bins, dtype=np.int64)
    edges = np.linspace(0.0, max_dist_m, n_bins + 1)
    centroid_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    n_dates = 0
    for _, day in df.groupby("date", sort=True):
        day = day[np.isfinite(day[value_col].to_numpy(dtype=float))]
        ids = day["cell_id"].unique()
        if len(ids) < 3:
            continue
        if len(ids) > k_cells:
            ids = np.sort(ids)[:k_cells]  # deterministic subsample
        key = tuple(sorted(ids))
        if key not in centroid_cache:
            centroid_cache[key] = cell_centroids(list(ids))
        lat, lon = centroid_cache[key]

        dists = _haversine_m(lat, lon)
        iu = np.triu_indices(len(ids), k=1)
        d = dists[iu]
        # day is already filtered to finite values, and ids come from it, so
        # the reindex-align is guaranteed NaN-free
        z = day.set_index("cell_id")[value_col].reindex(list(ids)).to_numpy(dtype=float)
        half2 = ((z[:, None] - z[None, :]) ** 2)[iu] / 2.0
        m = (d > 0.0) & (d <= max_dist_m)
        b = np.minimum(np.digitize(d[m], edges) - 1, n_bins - 1)
        np.add.at(gam_sum, b, half2[m])
        np.add.at(pair_cnt, b, 1)
        n_dates += 1

    if pair_cnt.sum() == 0:
        raise ValueError("no usable same-date pairs (need >=3 cells with values per date)")
    gamma = np.where(pair_cnt > 0, gam_sum / np.maximum(pair_cnt, 1), np.nan)
    centers = (edges[:-1] + edges[1:]) / 2.0
    ok = np.isfinite(gamma)
    sill = float(np.nanmax(gamma[ok]))
    rng, rmse, method = _fit_spherical(centers[ok], gamma[ok], sill)
    return SemivariogramResult(centers, gamma, pair_cnt, sill, rng, rmse, method, n_dates)


def _fit_spherical(h: np.ndarray, g: np.ndarray, sill: float) -> tuple[float, float, str]:
    """Least-squares exponential model gamma = s*(1 - exp(-h/r)), the shape
    atmospheric/plume residuals actually follow (steep near zero, long tail).
    Optimal s is closed-form per candidate r; the best RMSE wins. Falls back
    to the plan's 0.95-sill threshold rule when the fit is catastrophic."""
    best = None
    cand_r = np.linspace(max(h.min(), TROPOMI_CELL_M), h.max() * 4.0, 96)
    for r in cand_r:
        shape = 1.0 - np.exp(-h / r)
        s = float((g * shape).sum() / (shape * shape).sum())  # optimal s
        model = s * shape
        rmse = float(np.sqrt(np.mean((g - model) ** 2)))
        if best is None or rmse < best[1]:
            best = (float(r), rmse)
    r, rmse = best
    if rmse > 0.35 * sill or not np.isfinite(r):
        # catastrophic fit -> threshold rule: first lag reaching 95% of sill
        reach = np.nonzero(g >= 0.95 * sill)[0]
        r_thr = float(h[reach[0]]) if len(reach) else float(h.max())
        return r_thr, rmse, "threshold"
    return r, rmse, "exponential"


def advisory_block_size(result: SemivariogramResult, buffer: float = 1.0) -> float:
    """Plan section 6: spatial blocks larger than the autocorrelation range,
    with a buffer gap. `buffer` scales the gap between train/validation zones."""
    if buffer < 0:
        raise ValueError("buffer must be >= 0")
    return result.range_m * (1.0 + buffer)


def gate_g2_block_verdict(block_size_m: float, region_width_m: float) -> dict:
    """Plan G2: if the block size is larger than about half of Delhi's extent,
    spatial blocking is impractical -> chronological plus coarse blocks, and
    the spatial-generalization claim rests on leave-one-city-out only."""
    if region_width_m <= 0:
        raise ValueError("region_width_m must be positive")
    impractical = block_size_m > region_width_m / 2.0
    return {
        "block_size_m": float(block_size_m),
        "region_width_m": float(region_width_m),
        "blocking_feasible": not impractical,
        "protocol": "spatial_blocked"
        if not impractical
        else "chronological_plus_coarse_blocks; spatial claim via leave-one-city-out",
    }


def study_area_width_m(city: str, config_path: str | Path | None = None,
                       ) -> tuple[float, float]:
    """A city's bbox extent in metres (lon, lat) from the locked config.

    Phase 5 runs G2 per city, so the feasibility verdict must use THAT city's
    study-area width, not Delhi's. `config_path` defaults to the repo config;
    an unknown city raises, because silently scoring feasibility against the
    wrong rectangle would make the G2 advice mean something else.
    """
    from pathlib import Path as _P

    import yaml as _yaml

    cfg_path = (_P(config_path) if config_path is not None else
                _P(__file__).resolve().parents[2] / "configs" / "default.yaml")
    cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    sa = (cfg.get("study_areas") or {}).get(city)
    if not sa:
        raise KeyError(f"unknown study area {city!r}; have "
                       f"{sorted(cfg.get('study_areas') or {})}")
    lat0 = (float(sa["lat_min"]) + float(sa["lat_max"])) / 2.0
    dlon = ((float(sa["lon_max"]) - float(sa["lon_min"]))
            * np.cos(np.radians(lat0)))
    return (float(dlon * _EARTH_RADIUS_M * np.pi / 180.0),
            float((float(sa["lat_max"]) - float(sa["lat_min"]))
                  * _EARTH_RADIUS_M * np.pi / 180.0))


def ncr_width_m() -> tuple[float, float]:
    """Delhi-NCR bbox extent in metres (lat/lon), from the locked config bbox
    (76.38-78.62 E, 27.60-29.32 N); used by the G2 feasibility check."""
    return study_area_width_m("delhi_ncr")
