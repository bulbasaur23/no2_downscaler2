"""Static 0.05-degree grid and the observed-cell rule (decisions D2 and D8).

Why a module this early: the cell-ID scheme, grid geometry and observed-cell rule
are written in Phase 0 (G0) *before* any data export, so every later step -
training export, live polygons, cache keys, parity tests - shares one definition
of "cell". If this module is wrong, everything downstream is silently wrong.
"""
from __future__ import annotations

from dataclasses import dataclass

CELL_DEG = 0.05  # D8 - locked; TROPOMI native pixel is ~5.5 x 3.5 km.
ORIGIN_LAT = -90.0
ORIGIN_LON = -180.0  # grid origin so row/col indices are 0-based, stable, and sign-safe

_EARTH_RADIUS_M = 6371008.8  # IUGG mean radius used by Earth Engine's sf sphere


def _floor(v: float, quantum: float) -> int:
    # Round to 9 decimals first so exact multiples (e.g. lat = 76.05) do not
    # fall into the neighbouring cell because of float representation noise.
    v = round(v, 9)
    return int((v - 1e-9) // quantum) if abs(v / quantum - round(v / quantum)) < 1e-9 else int(v // quantum)


def row_from_lat(lat: float) -> int:
    """0-based row index counted up from the south pole."""
    return _floor(lat - ORIGIN_LAT, CELL_DEG)


def col_from_lon(lon: float) -> int:
    """0-based column index counted up from the antimeridian."""
    return _floor(lon - ORIGIN_LON, CELL_DEG)


def cell_id(row: int, col: int) -> str:
    """Stable, unique, human-readable ID: `r<row>c<col>` (5/6 digits zero-padded).

    Example: cells over Delhi are r1187c05195.. in row/col space; the padded form
    keeps lexicographic sorting stable, which matters for cache keys and parquet
    partitioning. Re-gridding the resolution changes the scheme -> bump "v" below.
    """
    return f"r{row:05d}c{col:06d}"


def parse_cell_id(cid: str) -> tuple[int, int]:
    """Inverse of :func:`cell_id`; raises on unknown formats."""
    if not (cid.startswith("r") and "c" in cid):
        raise ValueError(f"bad cell id: {cid!r}")
    r_part, c_part = cid[1:].split("c", 1)
    return int(r_part), int(c_part)


@dataclass(frozen=True)
class Cell:
    """One grid cell with its bounds (half-open: [min, max))."""

    id: str
    row: int
    col: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @property
    def center_lat(self) -> float:
        return (self.lat_min + self.lat_max) / 2.0

    @property
    def center_lon(self) -> float:
        return (self.lon_min + self.lon_max) / 2.0


def cell_from_point(lat: float, lon: float) -> Cell:
    row, col = row_from_lat(lat), col_from_lon(lon)
    return cell_from_row_col(row, col)


def cell_from_row_col(row: int, col: int) -> Cell:
    lat_min = ORIGIN_LAT + row * CELL_DEG
    lon_min = ORIGIN_LON + col * CELL_DEG
    return Cell(
        id=cell_id(row, col),
        row=row,
        col=col,
        lat_min=lat_min,
        lat_max=lat_min + CELL_DEG,
        lon_min=lon_min,
        lon_max=lon_min + CELL_DEG,
    )


def cells_in_bbox(lat_min: float, lat_max: float, lon_min: float, lon_max: float):
    """All cells that intersect the bbox (edge cells included - Phase 4 trims
    them by overlap fraction; here we list, weighting happens at summary time)."""
    row0, row1 = row_from_lat(lat_min), row_from_lat(max(lat_min, lat_max - 1e-12))
    col0, col1 = col_from_lon(lon_min), col_from_lon(max(lon_min, lon_max - 1e-12))
    for row in range(row0, row1 + 1):
        for col in range(col0, col1 + 1):
            yield cell_from_row_col(row, col)


# --------------------------------------------------------------------------
# Areas and overlap (needed for the observed-cell rule and edge-cell weights)
# --------------------------------------------------------------------------


def lat_band_area_m2(lat_min: float, lat_max: float, lon_width_deg: float) -> float:
    """Exact-area formula on a sphere for a constant-longitude patch:
    A = R^2 * (sin(phi2) - sin(phi1)) * dLambda. Using cos(lat)*area-of-band
    instead drifts a few percent at Indian latitudes; this does not."""
    import math

    p1, p2 = math.radians(lat_min), math.radians(lat_max)
    d_lam = math.radians(lon_width_deg)
    return _EARTH_RADIUS_M**2 * (math.sin(p2) - math.sin(p1)) * d_lam


def cell_area_m2(cell: Cell) -> float:
    return lat_band_area_m2(cell.lat_min, cell.lat_max, CELL_DEG)


def overlap_fraction_with_bbox(cell: Cell, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> float:
    """Fraction of the cell's *area* that lies inside the bbox.

    Area weighting matters: a cell 30% inside gets weight 0.3 when summaries are
    computed over a drawn polygon (plan Phase 4: edge cells keep an overlap
    fraction that weights the summary)."""
    i_lat_min, i_lat_max = max(cell.lat_min, lat_min), min(cell.lat_max, lat_max)
    i_lon_min, i_lon_max = max(cell.lon_min, lon_min), min(cell.lon_max, lon_max)
    if i_lat_min >= i_lat_max or i_lon_min >= i_lon_max:
        return 0.0
    overlap = lat_band_area_m2(i_lat_min, i_lat_max, i_lon_max - i_lon_min)
    return overlap / cell_area_m2(cell)


# --------------------------------------------------------------------------
# Observed-cell rule (D2)
# --------------------------------------------------------------------------


def observed_state(valid_area_fraction: float, main: float = 0.5) -> str:
    """D2: a cell is OBSERVED when at least `main` fraction of its area is
    covered by valid (QA-passed) pixels. Otherwise it is INSUFFICIENT with
    reason LOW_COVERAGE - never a value."""
    if not 0.0 <= valid_area_fraction <= 1.0:
        raise ValueError("valid_area_fraction must be in [0, 1]")
    return "observed" if valid_area_fraction >= main else "insufficient"


def observed_state_sensitivity(valid_area_fraction: float, thresholds=(0.30, 0.50, 0.70)) -> dict:
    """D2 sensitivity check: is the label stable across 30/50/70% thresholds?
    Divergence across thresholds is the early signal for risk 6."""
    return {f"{t:.2f}": observed_state(valid_area_fraction, main=t) for t in thresholds}
