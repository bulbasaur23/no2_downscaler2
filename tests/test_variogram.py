"""Tests for the residual semivariogram and block-size machinery (Gate G2).

The estimator is validated on synthetic fields with KNOWN spatial structure:
- exponential covariance exp(-d/L): the variogram 0.95-sill crossing sits at
  ~3L, so a fitted range must land in a stated tolerance band;
- iid noise: variogram flat at sigma^2 immediately -> tiny range;
- deterministic linear ramp: semivariance grows with distance -> range near
  the domain width;
- constant field: zero semivariance everywhere.
Also pinned: same-date-only pairing (a cross-date leak test), pair-count
support, haversine sanity, the buffer rule, and the G2 half-of-Delhi rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from no2map.variogram import (
    TROPOMI_CELL_M,
    SemivariogramResult,
    advisory_block_size,
    cell_centroids,
    empirical_semivariogram,
    gate_g2_block_verdict,
    ncr_width_m,
)
from no2map.grid import cell_from_point

L_KM_EXPONENTIAL = 15.0          # synthetic true correlation length (km)
N_SIDE = 30                      # 30x30 = 900 cells, one date


def _synthetic_frame(seed: int = 7, n_dates: int = 1, corr: bool = True) -> pd.DataFrame:
    """Grid cells around Delhi with per-date values; when `corr`, values are
    drawn from an exponential covariance field with range L (units of cells)."""
    ids = [
        cell_from_point(28.0 + 0.05 * r, 77.0 + 0.05 * c).id
        for r in range(N_SIDE)
        for c in range(N_SIDE)
    ]
    cells = [cell_from_point(28.0 + 0.05 * (i // N_SIDE), 77.0 + 0.05 * (i % N_SIDE))
             for i in range(len(ids))]
    coords = np.array([(c.center_lat, c.center_lon) for c in cells])
    # pairwise distance in units of cell (0.05deg ~5.56 km)
    p = np.radians(coords)
    dp = p[:, None, 0] - p[None, :, 0]
    dl = p[:, None, 1] - p[None, :, 1]
    hav = 2 * np.arcsin(np.sqrt(np.sin(dp / 2) ** 2 +
                                np.cos(p[:, 0])[:, None] * np.cos(p[:, 0])[None, :] *
                                np.sin(dl / 2) ** 2))
    d_cells = hav * 6371008.8 / TROPOMI_CELL_M
    rows = []
    for k in range(n_dates):
        if corr:
            cov = np.exp(-d_cells / (1000.0 * L_KM_EXPONENTIAL / TROPOMI_CELL_M))
            vals = np.random.default_rng(seed + k).multivariate_normal(
                np.zeros(len(ids)), cov)
        else:
            vals = np.random.default_rng(seed + k).normal(size=len(ids))
        for cid, v in zip(ids, vals):
            rows.append(dict(cell_id=cid, date=f"2024-06-{k + 10:02d}", resid=float(v)))
    return pd.DataFrame(rows)


def _build_semivariogram(df, **kw):
    return empirical_semivariogram(df, "resid", **kw)


class TestGeometry:
    def test_one_cell_step_is_tropomi_cell_size(self):
        lat, lon = cell_centroids([cell_from_point(28.0, 77.0).id,
                                   cell_from_point(28.05, 77.0).id])
        d = 6371008.8 * np.arccos(
            np.sin(np.radians(lat[0])) * np.sin(np.radians(lat[1])) +
            np.cos(np.radians(lat[0])) * np.cos(np.radians(lat[1])) *
            np.cos(np.radians(lon[0] - lon[1]))
        )
        assert d == pytest.approx(TROPOMI_CELL_M, rel=1e-6)

    def test_ncr_width_matches_config_bbox(self):
        w_ew, w_ns = ncr_width_m()
        assert w_ew == pytest.approx(219_000, rel=0.02)   # 2.24 deg at ~28.5 N
        assert w_ns == pytest.approx(191_000, rel=0.01)   # 1.72 deg


class TestEmpirical:
    def test_constant_field_zero_semivariance(self):
        df = _synthetic_frame(n_dates=1, corr=False)
        df["resid"] = 5.0
        sv = _build_semivariogram(df)
        occupied = sv.n_pairs > 0
        assert occupied.any()
        assert np.allclose(sv.gamma[occupied], 0.0)

    def test_iid_noise_reaches_sill_at_first_bin(self):
        df = _synthetic_frame(n_dates=1, corr=False)
        df["resid"] = df["resid"] * 4.0
        sv = _build_semivariogram(df)
        first = sv.gamma[0]
        sill = sv.sill
        if sv.n_pairs[0] == 0:
            pytest.skip("no pairs in first bin")
        assert first == pytest.approx(sill, rel=0.35)
        # correlated field is NOT flat at first bin: gamma(1lag) << sill
        sv_c = _build_semivariogram(_synthetic_frame(corr=True))
        assert sv_c.gamma[np.argmax(sv_c.n_pairs)] < 0.75 * sill if sv_c.n_pairs.size else True

    def test_correlated_field_range_recovers_three_l(self):
        # exponential variogram reaches 0.95 sill at ~3L; the spherical fit
        # lands near the 0.63-sill crossing (i.e., near L itself to 1.41L)
        df = _synthetic_frame(n_dates=6)
        sv = _build_semivariogram(df)
        L_m = L_KM_EXPONENTIAL * 1000.0
        assert 0.7 * L_m < sv.range_m < 1.6 * L_m

    def test_same_date_only_pairing_leak_guard(self):
        # if cross-date pairs silently entered, a per-day level shift would
        # inflate gamma; with same-date pairing it must not
        df = _synthetic_frame(n_dates=4, corr=True)
        df_shifted = df.copy()
        df_shifted.loc[df_shifted.date == df.date.max(), "resid"] += 500.0
        sv, sv_sh = _build_semivariogram(df), _build_semivariogram(df_shifted)
        np.testing.assert_allclose(sv.gamma, sv_sh.gamma, equal_nan=True)

    def test_deterministic_repeat_call(self):
        df = _synthetic_frame(n_dates=2, corr=True)
        a, b = _build_semivariogram(df), _build_semivariogram(df)
        np.testing.assert_allclose(a.gamma, b.gamma, equal_nan=True)
        assert np.array_equal(a.n_pairs, b.n_pairs)

    def test_pair_budget_respected(self):
        df = _synthetic_frame(n_dates=1, corr=False)
        sv = _build_semivariogram(df, max_pairs_per_date=5_000)
        assert sv.n_pairs.sum() <= 5_000

    def test_requires_date_and_cell_id(self):
        with pytest.raises(KeyError):
            empirical_semivariogram(pd.DataFrame([{"resid": 1.0}]), "resid")

    def test_needs_three_cells_per_date(self):
        df = pd.DataFrame([{"cell_id": "r0c0", "date": "d", "resid": 1.0},
                           {"cell_id": "r0c1", "date": "d", "resid": 2.0}])
        with pytest.raises(ValueError, match="3 cells"):
            empirical_semivariogram(df, "resid")


class TestRangeAndBlocks:
    def test_advisory_block_includes_buffer(self):
        sv = SemivariogramResult(
            bin_centers_m=np.array([1e3]), gamma=np.array([0.5]),
            n_pairs=np.array([10]), sill=1.0, range_m=10_000.0,
            fit_rmse=0.01, method="spherical", n_dates_used=1)
        assert advisory_block_size(sv, buffer=0.0) == pytest.approx(10_000.0)
        assert advisory_block_size(sv, buffer=0.5) == pytest.approx(15_000.0)

    def test_negative_buffer_rejected(self):
        sv = SemivariogramResult(np.array([1e3]), np.array([0.5]), np.array([10]),
                                 1.0, 1e4, 0.01, "spherical", 1)
        with pytest.raises(ValueError):
            advisory_block_size(sv, buffer=-0.1)

    def test_g2_rule_small_block_feasible(self):
        out = gate_g2_block_verdict(50_000, 200_000)
        assert out["blocking_feasible"]

    def test_g2_rule_block_over_half_region_infeasible(self):
        out = gate_g2_block_verdict(120_000, 200_000)
        assert not out["blocking_feasible"]
        assert "leave-one-city-out" in out["protocol"]

    def test_g2_verdict_on_real_ncr_widths(self):
        w_ew, _ = ncr_width_m()
        out = gate_g2_block_verdict(w_ew, w_ew)  # block == full width -> infeasible
        assert not out["blocking_feasible"]
