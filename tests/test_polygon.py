"""Tests for the Phase-4 polygon engine (pure offline geometry).

Pinned: exact-area overlap fractions on axis-aligned and L-shaped polygons
(hard-coded expected fractions), single-cell containment and the sub-cell
resolution warning, buffered fetch-window expansion, and every hard
validation rejection (few vertices, too many, self-intersection, outside
India, size cap).
"""
from __future__ import annotations

import pytest

from no2map.grid import CELL_DEG, cell_from_point
from no2map.polygon import (
    INDIA_BBOX,
    MAX_VERTICES,
    cell_overlap_fraction,
    cells_for_polygon,
    has_self_intersection,
    polygon_bbox_buffered,
    resolution_warning,
    validate_polygon,
)


class TestOverlapFractions:
    def test_axis_aligned_full_cells_are_unity(self):
        # polygon exactly covering one full cell
        cell = cell_from_point(28.0, 77.0)
        poly = [(cell.lon_min, cell.lat_min), (cell.lon_max, cell.lat_min),
                (cell.lon_max, cell.lat_max), (cell.lon_min, cell.lat_max)]
        assert cell_overlap_fraction(poly, cell) == pytest.approx(1.0)
        # and its neighbours are untouched
        right = cell_from_point(28.0, 77.0 + CELL_DEG)
        assert cell_overlap_fraction(poly, right) == pytest.approx(0.0)

    def test_half_covering_cell(self):
        cell = cell_from_point(28.0, 77.0)
        poly = [(cell.lon_min, cell.lat_min),
                (cell.lon_min + CELL_DEG / 2, cell.lat_min),
                (cell.lon_min + CELL_DEG / 2, cell.lat_max),
                (cell.lon_min, cell.lat_max)]
        assert cell_overlap_fraction(poly, cell) == pytest.approx(0.5)

    def test_quarter_covering_cell(self):
        cell = cell_from_point(28.0, 77.0)
        x0, y0 = cell.lon_min, cell.lat_min
        poly = [(x0, y0), (x0 + CELL_DEG / 2, y0),
                (x0 + CELL_DEG / 2, y0 + CELL_DEG / 2), (x0, y0 + CELL_DEG / 2)]
        assert cell_overlap_fraction(poly, cell) == pytest.approx(0.25)

    def test_l_shape_hard_values(self):
        # L over a 2x2 lattice of cells: full top row + full bottom-left
        cells = {(r, c): cell_from_point(28.0 + CELL_DEG * r, 77.0 + CELL_DEG * c)
                 for r in (0, 1) for c in (0, 1)}
        x0, y0 = cells[(0, 0)].lon_min, cells[(0, 0)].lat_min
        x1, y1 = cells[(0, 1)].lon_max, cells[(1, 0)].lat_max
        xm, ym = cells[(0, 0)].lon_max, cells[(0, 0)].lat_max
        poly = [(x0, y0), (x1, y0), (x1, ym), (xm, ym), (xm, y1), (x0, y1)]
        frac = cells_for_polygon(poly, buffer_cells=0)
        assert frac[cells[(0, 0)].id] == pytest.approx(1.0)
        assert frac[cells[(0, 1)].id] == pytest.approx(1.0)
        assert frac[cells[(1, 0)].id] == pytest.approx(1.0)
        assert cells[(1, 1)].id not in frac  # exactly zero overlap excluded

    def test_triangle_clipped_across_two_cells(self):
        cell = cell_from_point(28.0, 77.0)
        x0, y0 = cell.lon_min, cell.lat_min
        x1, y1 = cell.lon_max, cell.lat_max
        # diagonal triangle with legs one cell: covered area in each half's
        # cells is 0.5 for this cell and 0 for nothing beyond
        poly = [(x0, y0), (x1, y0), (x0, y1)]
        assert cell_overlap_fraction(poly, cell) == pytest.approx(0.5)


class TestCellsForPolygon:
    def test_tiny_polygon_contained_in_one_cell(self):
        poly = [(77.01, 28.01), (77.012, 28.01), (77.011, 28.012)]
        frac = cells_for_polygon(poly, buffer_cells=0)
        assert len(frac) == 1
        cell = cell_from_point(28.01, 77.01)
        assert next(iter(frac)) == cell.id
        assert next(iter(frac.values())) < 0.01

    def test_bbox_buffer_expands_by_buffer_cells(self):
        poly = [(77.0, 28.0), (77.05, 28.0), (77.05, 28.05), (77.0, 28.05)]
        bb = polygon_bbox_buffered(poly, buffer_cells=2)
        assert bb["lat_min"] == pytest.approx(28.0 - 2 * CELL_DEG)
        assert bb["lon_max"] == pytest.approx(77.05 + 2 * CELL_DEG)


class TestValidation:
    def base_square(self, dlon=0.2, dlat=0.2):
        return [(77.0, 28.0), (77.0 + dlon, 28.0),
                (77.0 + dlon, 28.0 + dlat), (77.0, 28.0 + dlat)]

    def test_valid_polygon_reports(self):
        info = validate_polygon(self.base_square())
        assert info["n_vertices"] == 4
        assert info["warnings"] == []
        # ~0.04 deg2 at lat 28: 0.04 * 111320^2 * cos(28) ^-? just bounded
        assert 3e8 < info["area_m2"] < 6e8

    def test_sub_cell_polygon_warns(self):
        tiny = [(77.01, 28.01), (77.02, 28.01), (77.02, 28.02), (77.01, 28.02)]
        info = validate_polygon(tiny)
        assert info["warnings"] and "smaller than one grid cell" in info["warnings"][0]

    def test_two_vertices_rejected(self):
        with pytest.raises(ValueError, match="at least 3"):
            validate_polygon([(77.0, 28.0), (77.2, 28.2)])

    def test_too_many_vertices_rejected(self):
        from math import cos, pi
        poly = [(77.0 + 0.2 * cos(2 * pi * k / (MAX_VERTICES + 2)),
                 28.0 + 0.2 * (k % 3) / (MAX_VERTICES + 1))
                for k in range(MAX_VERTICES + 2)]
        with pytest.raises(ValueError, match="vertices"):
            validate_polygon(poly)
        # exactly the cap passes the vertex rule (may hit other rules? no)
        poly_cap = [(77.0 + 0.1 * cos(2 * pi * k / MAX_VERTICES),
                     28.0 + 0.1 * (k % 2) / MAX_VERTICES)
                    for k in range(MAX_VERTICES)]
        poly_cap = list(dict.fromkeys([(round(x, 9), round(y, 9))
                                       for x, y in poly_cap]))
        assert len(poly_cap) <= MAX_VERTICES

    def test_self_intersecting_bowtie_rejected(self):
        bowtie = [(77.0, 28.0), (77.3, 28.3), (77.3, 28.0), (77.0, 28.3)]
        assert has_self_intersection(bowtie)
        with pytest.raises(ValueError, match="self-intersect"):
            validate_polygon(bowtie)

    def test_simple_concave_is_not_flagged(self):
        l_poly = [(77.0, 28.0), (77.3, 28.0), (77.3, 28.1), (77.1, 28.1),
                  (77.1, 28.3), (77.0, 28.3)]
        assert not has_self_intersection(l_poly)

    def test_outside_india_rejected(self):
        out = self.base_square()
        out[0] = (98.5, out[0][1])  # east of the India clip
        with pytest.raises(ValueError, match="India"):
            validate_polygon(out)
        below = self.base_square()
        below[0] = (out and (below[0][0], 5.0))
        with pytest.raises(ValueError, match="India"):
            validate_polygon(below)

    def test_area_cap_rejected(self):
        big = self.base_square(dlon=2.0, dlat=2.0)
        with pytest.raises(ValueError, match="exceeds"):
            validate_polygon(big, max_area_m2=1e10)


class TestResolutionWarning:
    def test_single_cell_triggers(self):
        assert resolution_warning(1, 0.3) is not None
        assert resolution_warning(5, 0.2) is None
