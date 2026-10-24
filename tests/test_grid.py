"""Grid scheme and observed-cell rule tests (D2, D8)."""
import pytest

from no2map.grid import (
    CELL_DEG,
    Cell,
    cell_area_m2,
    cell_from_point,
    cell_id,
    cells_in_bbox,
    observed_state,
    observed_state_sensitivity,
    overlap_fraction_with_bbox,
    parse_cell_id,
)


class TestCellScheme:
    def test_known_point_maps_to_expected_cell(self):
        # Delhi (28.61 N, 77.21 E): lat bucket [28.60, 28.65), lon bucket [77.20, 77.25)
        c = cell_from_point(28.61, 77.21)
        assert c.lat_min == pytest.approx(28.60, abs=1e-9)
        assert c.lat_max == pytest.approx(28.65, abs=1e-9)
        assert c.lon_min == pytest.approx(77.20, abs=1e-9)
        assert c.lon_max == pytest.approx(77.25, abs=1e-9)

    def test_grid_origin_is_stable_and_signsafe(self):
        south = cell_from_point(-89.99, -179.99)
        assert south.lat_min == pytest.approx(-90.0)
        assert south.lon_min == pytest.approx(-180.0)

    def test_round_trip_id(self):
        c = cell_from_point(13.0, 77.6)  # Bengaluru
        row, col = parse_cell_id(c.id)
        assert cell_id(row, col) == c.id

    def test_ids_are_unique_per_cell(self):
        a = cell_from_point(28.61, 77.21)
        b = cell_from_point(28.63, 77.23)
        assert a.id == b.id  # same cell bucket
        assert a.id != cell_from_point(28.66, 77.21).id  # next lat bucket differs

    def test_contiguous_points_neighbour(self):
        a = cell_from_point(28.61, 77.21)
        b = cell_from_point(28.61, 77.26)
        assert b.col - a.col == 1

    def test_parse_rejects_bad_ids(self):
        with pytest.raises(ValueError):
            parse_cell_id("nope")


class TestGeometry:
    def test_cell_area_is_sane(self):
        # a 0.05 deg cell at lat 28 deg ~ (0.05*111.32) x (0.05*111.13*cos28) km^2 ~ 27.1 km^2
        c = cell_from_point(28.0, 77.0)
        area = cell_area_m2(c)
        assert 2.4e7 < area < 3.0e7

    def test_full_overlap_is_one(self):
        c = cell_from_point(28.61, 77.21)
        f = overlap_fraction_with_bbox(c, c.lat_min, c.lat_max, c.lon_min, c.lon_max)
        assert f == pytest.approx(1.0)

    def test_half_overlap(self):
        c = cell_from_point(28.61, 77.21)
        f = overlap_fraction_with_bbox(c, c.lat_min, c.lat_max, 77.225, c.lon_max)
        assert f == pytest.approx(0.5, rel=1e-3)

    def test_no_overlap_is_zero(self):
        c = cell_from_point(28.61, 77.21)
        f = overlap_fraction_with_bbox(c, 28.0, 28.2, 80.0, 80.2)
        assert f == 0.0

    def test_bbox_cell_count_ncr(self):
        # Expected count derives from the SAME index math (single definition of
        # "which cells does a bbox touch"), then sanity-bounds the NCR region.
        from no2map.grid import col_from_lon, row_from_lat

        cells = list(cells_in_bbox(27.60, 29.32, 76.38, 78.62))
        n_rows = row_from_lat(29.32) - row_from_lat(27.60) + 1
        n_cols = col_from_lon(78.62) - col_from_lon(76.38) + 1
        assert len(cells) == n_rows * n_cols
        assert 1400 < len(cells) < 1800  # NCR-bbox ballpark; reported to feasibility note


class TestObservedRule:
    @pytest.mark.parametrize("frac,expected", [(0.0, "insufficient"), (0.49, "insufficient"), (0.50, "observed"), (0.9, "observed")])
    def test_main_threshold(self, frac, expected):
        assert observed_state(frac) == expected

    def test_reject_invalid_fraction(self):
        with pytest.raises(ValueError):
            observed_state(1.5)
        with pytest.raises(ValueError):
            observed_state(-0.1)

    def test_sensitivity_dictionary(self):
        out = observed_state_sensitivity(0.45)
        assert out == {"0.30": "observed", "0.50": "insufficient", "0.70": "insufficient"}
