"""Static per-cell features (Phase 1): elevation statistics and water fraction.

Computed ONCE per city (they do not vary by date) and cached to a Parquet file
that the monthly exports join on cell_id. Concept: plan section 4 - roads /
elevation / emitters exist at ~1 km resolution and are SUMMARISED per
0.05-deg cell (D8); roads and emitters arrive in Phase 2 through OSM/osmnx,
so Phase 1 ships elevation + water today (both Earth-Engine-side, cheap).

Water fraction uses ESA WorldCover v200 class 80 (permanent water) because the
JRC GSW occurrence raster was verified broken on GEE (feasibility anomaly A2).
"""
from __future__ import annotations

import ee

WORLD_COVER = "ESA/WorldCover/v200/2021"
SRTM = "USGS/SRTMGL1_003"

WATER_CLASS = 80  # permanent water bodies

STATIC_COLUMNS = [
    "elevation_mean_m", "elevation_std_m", "elevation_min_m", "elevation_max_m",
    "water_fraction", "water_state",
]


def _water_state(water_fraction: float | None, land_cut: float, water_cut: float) -> str | None:
    """land/mixed/water classification from configured cutoffs (risk 15)."""
    if water_fraction is None:
        return None
    if water_fraction <= land_cut:
        return "land"
    if water_fraction > water_cut:
        return "water"
    return "mixed"


def static_features_per_cell(cells_fc: ee.FeatureCollection, cfg: dict) -> list[dict]:
    """Two reduceRegions per city: elevation stats and water fraction."""
    elev_img = ee.Image(SRTM).select("elevation")
    elev = elev_img.reduceRegions(
        collection=cells_fc,
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev().combine(
            ee.Reducer.minMax(), sharedInputs=True), sharedInputs=True),
        scale=90,
        tileScale=4,
    ).getInfo()["features"]

    water = (
        ee.Image(WORLD_COVER).select("Map").eq(WATER_CLASS)
    )
    water_rows = water.reduceRegions(
        collection=cells_fc,
        reducer=ee.Reducer.mean(),
        scale=100,
        tileScale=4,
    ).getInfo()["features"]

    elev_by_cell = {
        f["properties"]["cell_id"]: {
            "elevation_mean_m": f["properties"].get("mean"),
            "elevation_std_m": f["properties"].get("stdDev"),
            "elevation_min_m": f["properties"].get("min"),
            "elevation_max_m": f["properties"].get("max"),
        }
        for f in elev
    }
    water_by_cell = {
        f["properties"]["cell_id"]: f["properties"].get("mean")
        for f in water_rows
    }

    cuts = cfg["water_cutoffs"]
    out = []
    for cell_id, e in elev_by_cell.items():
        w = water_by_cell.get(cell_id)
        out.append(
            {
                "cell_id": cell_id,
                **{k: (round(v, 4) if v is not None else None) for k, v in e.items()},
                "water_fraction": round(float(w), 6) if w is not None else None,
                "water_state": _water_state(w, cuts["land_max_water_fraction"], cuts["water_min_water_fraction"]),
            }
        )
    return out
