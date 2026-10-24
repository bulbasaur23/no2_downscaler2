"""Polygon geometry for the live path (Phase 4).

Everything the drawn-polygon flow needs BEFORE any Earth-Engine call, pure
python and fully testable offline:

- validate_polygon: >=3 vertices, <=40, self-intersection (O(n^2) segment
  test), India bounding-clip, optional size cap (the plan ties the cap to
  the quota measurement - the caller supplies it), auto-closes rings.
- cells_for_polygon: every grid cell (global 0.05 lattice) whose area the
  polygon touches, each with EXACT overlap fraction via Sutherland-Hodgman
  clipping of the polygon by the cell rectangle, area-weighted consistently
  with the cell's own area (Phase 4: edge cells keep an overlap fraction
  that weights the summary).
- polygon_bbox_buffered: the compute bbox (polygon bbox + `buffer_cells`,
  2-3 planned) used as the fetch window - results are returned for cells
  inside the polygon only, never for the buffer ring.
- resolution_warning: single home for the sub-cell warning wording.

Only fractions > 0 are returned; a polygon smaller than one cell yields a
single cell with a small fraction.
"""
from __future__ import annotations

import math

from no2map.grid import CELL_DEG, Cell, cell_from_row_col, col_from_lon, row_from_lat

M_PER_DEG_LAT = 6371008.8 * math.pi / 180.0  # same value the grid/splits use

INDIA_BBOX = (6.0, 37.5, 68.0, 97.5)  # lat_min, lat_max, lon_min, lon_max
MAX_VERTICES = 40

Polygon = list[tuple[float, float]]  # (lon, lat) ring, open or closed


# ---- basic plane geometry ------------------------------------------------

def _dedupe_close(poly: Polygon) -> Polygon:
    out: Polygon = []
    for p in poly:
        if not out or math.dist(p, out[-1]) > 1e-12:
            out.append((float(p[0]), float(p[1])))
    if len(out) > 1 and math.dist(out[0], out[-1]) <= 1e-12:
        out.pop()
    return out


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(p1, p2, q1, q2) -> bool:
    """Proper crossing test. Endpoint-only sharing (a legal polygon edge) is
    excluded; collinear overlap runs through the d1..d4 sign test."""
    d1 = _orient(q1, q2, p1)
    d2 = _orient(q1, q2, p2)
    d3 = _orient(p1, p2, q1)
    d4 = _orient(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def has_self_intersection(poly: Polygon) -> bool:
    n = len(poly)
    for i in range(n):
        a1, a2 = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            if (j == i + 1) or (i == 0 and j == n - 1):
                continue  # adjacent edges share a vertex legitimately
            if _segments_cross(a1, a2, poly[j], poly[(j + 1) % n]):
                return True
    return False


def _shoelace_scaled(poly: Polygon, coslat: float) -> float:
    """Planar area with lon scaled by cos(lat), consistent with the planar
    cell-area used as the weighting denominator. Error at Indian latitudes
    is far below the D6 threshold granularity (<1%)."""
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i][0] * coslat, poly[i][1]
        x2, y2 = poly[(i + 1) % n][0] * coslat, poly[(i + 1) % n][1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


# ---- Sutherland-Hodgman clipping by the cell rectangle --------------------

def clip_polygon_by_cell(poly: Polygon, cell: Cell) -> Polygon:
    """Polygon (lon,lat) clipped to the cell's lat/lon rectangle."""
    pts = poly
    # left (lon >= lon_min), right (lon <= lon_max), bottom, top
    pts = _clip(pts, 0, cell.lon_min, True)
    pts = _clip(pts, 0, cell.lon_max, False)
    pts = _clip(pts, 1, cell.lat_min, True)
    pts = _clip(pts, 1, cell.lat_max, False)
    return pts


def _clip(pts: Polygon, axis: int, bound: float, keep_ge: bool) -> Polygon:
    """One Sutherland-Hodgman half-plane clip. Keeps axis <op> bound."""
    def inside(p) -> bool:
        v = p[axis]
        return v >= bound if keep_ge else v <= bound

    def crossing(a, b) -> tuple[float, float]:
        t = (bound - a[axis]) / (b[axis] - a[axis])
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        return (x, y)

    out: list = []
    if not pts:
        return out
    prev, prev_in = pts[-1], inside(pts[-1])
    for cur in pts:
        cur_in = inside(cur)
        if cur_in:
            if not prev_in:
                out.append(crossing(prev, cur))
            out.append(cur)
        elif prev_in:
            out.append(crossing(prev, cur))
        prev, prev_in = cur, cur_in
    return out


# ---- public API -----------------------------------------------------------

def cell_overlap_fraction(poly: Polygon, cell: Cell) -> float:
    """Fraction of the CELL's area inside the polygon (0 if disjoint)."""
    clipped = clip_polygon_by_cell(poly, cell)
    if len(clipped) < 3:
        return 0.0
    coslat = math.cos(math.radians((cell.lat_min + cell.lat_max) / 2.0))
    num = _shoelace_scaled(clipped, coslat)
    denom = CELL_DEG * coslat * CELL_DEG  # planar cell area (same scaling)
    return max(0.0, min(1.0, num / denom))


def validate_polygon(
    poly: Polygon,
    max_area_m2: float | None = None,
    m_per_deg_lat: float = M_PER_DEG_LAT,
) -> dict:
    """Raise ValueError on hard violations; return {"area_m2", "n_vertices",
    "warnings"} otherwise. `warnings` includes the sub-cell resolution note
    when the polygon is smaller than one 0.05 cell."""
    pts = _dedupe_close(list(poly))
    if len(pts) < 3:
        raise ValueError("polygon needs at least 3 distinct vertices")
    if len(pts) > MAX_VERTICES:
        raise ValueError(f"polygon has {len(pts)} vertices; max {MAX_VERTICES}")
    lat = [p[1] for p in pts]
    lon = [p[0] for p in pts]
    i_lat0, i_lat1 = INDIA_BBOX[0], INDIA_BBOX[1]
    i_lon0, i_lon1 = INDIA_BBOX[2], INDIA_BBOX[3]
    if max(lat) > i_lat1 or min(lat) < i_lat0 or max(lon) > i_lon1 or min(lon) < i_lon0:
        raise ValueError("polygon extends outside the study region "
                         "(India bounding box)")
    if has_self_intersection(pts):
        raise ValueError("polygon is self-intersecting")

    warnings: list[str] = []
    mid_lat = (min(lat) + max(lat)) / 2.0
    coslat = math.cos(math.radians(mid_lat))
    area_deg2 = _shoelace_scaled(pts, coslat)
    area_m2 = area_deg2 * (m_per_deg_lat ** 2) * coslat
    if max_area_m2 is not None and area_m2 > max_area_m2:
        raise ValueError(
            f"polygon area {area_m2 / 1e6:.1f} km2 exceeds the limit "
            f"{max_area_m2 / 1e6:.1f} km2")
    cell_area = (CELL_DEG ** 2) * (m_per_deg_lat ** 2) * coslat
    if area_m2 < cell_area:
        warnings.append(
            "polygon is smaller than one grid cell (about 5.5 x 3.5 km); "
            "returning the containing cell at native resolution")
    return {"area_m2": area_m2, "n_vertices": len(pts), "warnings": warnings}


def cells_for_polygon(poly: Polygon, buffer_cells: int = 2) -> dict[str, float]:
    """cell_id -> overlap fraction for every cell the polygon touches."""
    pts = _dedupe_close(list(poly))
    lat = [p[1] for p in pts]
    lon = [p[0] for p in pts]
    out: dict[str, float] = {}
    r0, r1 = row_from_lat(min(lat)) - 1, row_from_lat(max(lat)) + 1
    c0, c1 = col_from_lon(min(lon)) - 1, col_from_lon(max(lon)) + 1
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            cell = cell_from_row_col(r, c)
            f = cell_overlap_fraction(pts, cell)
            if f > 1e-9:
                out[cell.id] = f
    return out


def polygon_bbox_buffered(poly: Polygon, buffer_cells: int = 2) -> dict:
    """Lat/lon bbox of the polygon expanded by `buffer_cells` (the Phase-4
    fetch window: compute on bbox + buffer, return polygon cells only)."""
    pts = _dedupe_close(list(poly))
    lat = [p[1] for p in pts]
    lon = [p[0] for p in pts]
    pad = buffer_cells * CELL_DEG
    return {
        "lat_min": min(lat) - pad, "lat_max": max(lat) + pad,
        "lon_min": min(lon) - pad, "lon_max": max(lon) + pad,
    }


def resolution_warning(n_cells: int, min_fraction: float) -> str | None:
    if n_cells == 1 or (n_cells and min_fraction < 0.5 and n_cells <= 4):
        return ("region is small relative to the 0.05 grid; summary uses "
                "one or few partially covered cells at native resolution "
                "(about 5.5 x 3.5 km)")
    return None
