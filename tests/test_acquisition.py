"""Phase-1 acquisition tests: pure D2 arithmetic, overpass windows, dates.

No Earth Engine calls here - the GEE side is exercised by real export runs
and later by the parity test; these tests pin the math that decides which
cells are OBSERVED and what the target value means (risk 6/18)."""
import pytest

from no2map.acquisition import (
    MOL_TO_UMOL,
    cell_stats_from_sums,
    day_rows,
    month_days,
    no_overpass_rows,
    overpass_utc_hours,
)


class TestOverpassWindows:
    def test_delhi_window_spans_drift(self):
        # h0 = 13.5 - 77.5/15 = 8.3667; envelope buckets 8,9 widened by the
        # +/-1 drift headroom (measured: a [8,9] window caught only 6/31 days)
        assert overpass_utc_hours(76.38, 78.62) == [7, 8, 9, 10]

    def test_bengaluru_window(self):
        # lon ~77.6: h0 = 8.3267 -> buckets 8,9 widened -> 7..10
        assert overpass_utc_hours(77.30, 77.90) == [7, 8, 9, 10]

    def test_kolkata_window(self):
        # lon ~88.2: h0 = 7.62 -> buckets 7,8 widened -> 6..9
        assert overpass_utc_hours(87.90, 88.50) == [6, 7, 8, 9]

    def test_window_matches_solar_noon_algebra(self):
        # lon_center = 0: h0 = 13.5 -> buckets 13,14 widened -> 12..15
        assert overpass_utc_hours(-1.0, 1.0) == [12, 13, 14, 15]


class TestMonthDays:
    def test_31_day_month(self):
        d = month_days("2025-01")
        assert d[0] == "2025-01-01" and d[-1] == "2025-01-31" and len(d) == 31

    def test_28_day_month(self):
        assert len(month_days("2025-02")) == 28
        assert month_days("2024-02")[-1] == "2024-02-29"  # leap year


class TestCellStatsD2:
    def test_perfect_observation(self):
        v = 100e-5  # per-pixel mol/m^2 (same value everywhere -> zero spread)
        out = cell_stats_from_sums(
            value_sum=v * 100,
            value_sq_sum=v * v * 100,  # 100 identical pixels,
            valid_sum=100, one_sum=100, orbit_count_sum=150,
        )
        assert out["state"] == "observed"
        assert out["n_valid_fraction"] == 1.0
        assert out["subpixel_spread_umol_m2"] == 0.0
        assert out["n_orbit_overlap"] == 1.5
        expected = 100e-5 * MOL_TO_UMOL  # 1000 umol/m2
        assert out["value_umol_m2"] == pytest.approx(expected, rel=1e-3)

    def test_half_coverage_threshold_boundary(self):
        # exactly 50% valid -> observed (>= main threshold, D2)
        out = cell_stats_from_sums(50e-5, (50e-5) ** 2 * 50, 50, 100, 50)
        assert out["n_valid_fraction"] == 0.5
        assert out["state"] == "observed"

    def test_under_threshold_is_insufficient_with_reason(self):
        out = cell_stats_from_sums(30e-5, (30e-5) ** 2 * 30, 30, 100, 30)
        assert out["state"] == "insufficient"
        assert out["reason_code"] == "low_coverage"

    def test_zero_valid_is_no_overpass_not_insufficient(self):
        # D6: a cell with no orbit data gets no_overpass, NOT insufficient
        out = cell_stats_from_sums(0.0, 0.0, 0, 100, 0)
        assert out["state"] == "no_overpass"
        assert out["reason_code"] == "no_overpass"
        assert out["value_umol_m2"] is None
        assert out["n_valid_fraction"] == 0.0

    def test_subpixel_spread_positive(self):
        # 10 valid pixels with values 0..9 e-5 -> spread > 0
        vals = [v * 1e-5 for v in range(10)]
        out = cell_stats_from_sums(
            value_sum=sum(vals),
            value_sq_sum=sum(v * v for v in vals),
            valid_sum=10, one_sum=10, orbit_count_sum=10,
        )
        assert out["subpixel_spread_umol_m2"] > 0
        assert out["value_umol_m2"] == pytest.approx(sum(vals) / 10 * MOL_TO_UMOL, rel=1e-6)

    def test_unit_conversion_applies(self):
        mean_mol = 4.5e-5  # measured NCR monthly mean magnitude
        out = cell_stats_from_sums(mean_mol * 50, mean_mol**2 * 50, 50, 50, 50)
        assert out["value_umol_m2"] == pytest.approx(45.0, abs=0.1)  # x1e6 (D1)

    def test_rejects_zero_total(self):
        with pytest.raises(ValueError):
            cell_stats_from_sums(1.0, 1.0, 1, 0, 1)


class TestDayRows:
    """One row per (cell, date) even when the date has no overpass (D6)."""

    CELLS = ["7.0_76.0", "7.0_76.1", "7.1_76.0"]

    def test_empty_day_becomes_no_overpass_rows_for_every_cell(self):
        # regression: an empty overpass window used to return [] and then a
        # bandless composite crashed the whole month export on .pow(2)
        rows = day_rows([], self.CELLS, main_threshold=0.5)
        assert [r["cell_id"] for r in rows] == self.CELLS
        for r in rows:
            assert r["state"] == "no_overpass"
            assert r["reason_code"] == "no_overpass"
            assert r["value_umol_m2"] is None
            assert r["n_valid_fraction"] == 0.0
            assert r["n_orbit_overlap"] == 0.0

    def test_no_overpass_rows_helper_is_pure(self):
        # called twice -> equal rows, no shared mutable state
        a = no_overpass_rows(self.CELLS)
        b = no_overpass_rows(self.CELLS)
        assert a == b
        a[0]["state"] = "mutated"
        assert no_overpass_rows(self.CELLS)[0]["state"] == "no_overpass"

    def test_non_empty_day_uses_d2_arithmetic_unchanged(self):
        sums = [{
            "cell_id": "7.0_76.0", "value_sum": 50e-5,
            "value_sq_sum": (50e-5) ** 2 * 50, "valid_sum": 50.0,
            "one_sum": 100.0, "orbit_count_sum": 50.0,
        }]
        rows = day_rows(sums, self.CELLS, main_threshold=0.5)
        assert len(rows) == 1  # only cells the fetcher returned
        assert rows[0]["state"] == "observed"
        assert rows[0]["value_umol_m2"] == pytest.approx(10.0, rel=1e-3)

    def test_day_rows_respects_threshold_argument(self):
        sums = [{
            "cell_id": "c", "value_sum": 30e-5,
            "value_sq_sum": (30e-5) ** 2 * 30, "valid_sum": 30.0,
            "one_sum": 100.0, "orbit_count_sum": 30.0,
        }]
        assert day_rows(sums, ["c"], main_threshold=0.5)[0]["state"] == "insufficient"
        assert day_rows(sums, ["c"], main_threshold=0.2)[0]["state"] == "observed"
