"""Weather acquisition (Phase 1, decisions D4 and D8).

Concept: the same per-(cell, date) keys as the NO2 rows, but for weather -
matched to the overpass using its UTC-hour window (D4 timezone rule), plus the
lagged features from D8: previous-day and 48h mean wind (10 m and 100 m),
lagged precipitation, boundary-layer height at the overpass hour.

Measured Stage-0 facts encoded here (docs/feasibility.md):
 - ERA5-Land has temperature, dewpoint, surface pressure, 10 m wind and
   precip but NO 100 m wind and NO BLH; those come from ECMWF/ERA5/HOURLY.
 - Both collections are slightly behind real time (~6 days as of measure);
   the app date cap uses the earliest of the two (D4).

Window definitions (all UTC):
 - inst   : overpass window [day 07:00, day 10:00) - means, not instants, to
            damp alignment noise, per D4's 'matched to the overpass hour',
 - lag24  : [day-1 08:00, day 08:00) mean wind (D8), precipitation SUM of the
            de-accumulated hourly band (total_precipitation_hourly; the raw
            band resets at midnight and would over-count across the reset),
 - lag48  : [day-2 08:00, day 08:00) mean wind, precipitation SUM.
Lags use ONLY weather (never NO2), so live requests see the same inputs.

Weather band naming (verified by Stage-0 probe g0_weather.json):
 ERA5-Land: u_component_of_wind_10m, v_component_of_wind_10m, temperature_2m,
 dewpoint_temperature_2m, surface_pressure, total_precipitation.
 ERA5/HOURLY single levels: u/v_component_of_wind_10m,
 u/v_component_of_wind_100m, boundary_layer_height.

 Precipitation caveat (verified against the GEE catalog): ERA5-Land's
 `total_precipitation` ACCUMULATES from the start of the day's forecast and
 RESETS AT MIDNIGHT, so summing it over a lag window crossing midnight
 over-counts. The catalog's `total_precipitation_hourly` band is the
 de-accumulated hourly depth - exactly what lag sums need; LAND_PRECIP_BAND
 uses it.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import ee

ERA5_LAND = "ECMWF/ERA5_LAND/HOURLY"
ERA5_HOURLY = "ECMWF/ERA5/HOURLY"

LAND_INST_BANDS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "surface_pressure",
]
# The hourly band is de-accumulated by GEE (difference of consecutive forecast
# steps of the raw, midnight-resetting cumulative band). Safe to SUM across
# midnight; the raw `total_precipitation` band is NOT.
LAND_PRECIP_BAND = "total_precipitation_hourly"  # meters per hour
# HOURLY single-levels bands used for instant + lags:
SINGLE_WIND_BANDS = [
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "u_component_of_wind_100m",
    "v_component_of_wind_100m",
]
SINGLE_BLH_BAND = "boundary_layer_height"

# Export column names (shared with the parquet schema / data dictionary)
WEATHER_COLUMNS = [
    "t2m_k_inst", "d2m_k_inst", "sp_pa_inst",
    "wind_u10_inst", "wind_v10_inst", "wind_u100_inst", "wind_v100_inst", "blh_m_inst",
    "wind_u10_l24", "wind_v10_l24", "wind_u100_l24", "wind_v100_l24",
    "wind_u10_l48", "wind_v10_l48", "wind_u100_l48", "wind_v100_l48",
    "precip_l24_m", "precip_l48_m",
]


def _stamp(date_str: str, hour: int) -> str:
    return f"{date_str}T{hour:02d}:00:00"


def _shift(date_str: str, days: int) -> str:
    d = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)
    return d.strftime("%Y-%m-%d")


def day_weather_stack(date_str: str, scale_m: float, cells_fc: ee.FeatureCollection) -> list[dict]:
    """One reduceRegions per day: all weather bands for every cell at once.

    Band name -> export column mapping happens here (_SUFFIX tables), so the
    Parquet schema is defined in one place and the phase-4 live path reuses
    this exact function (the training-serving parity guarantee, risk 10)."""
    inst_start, inst_end = _stamp(date_str, 7), _stamp(date_str, 10)

    # --- single levels (100 m wind, BLH) -----------------------------------
    single = ee.ImageCollection(ERA5_HOURLY)
    inst_s = (
        single.filterDate(inst_start, inst_end)
        .select(SINGLE_WIND_BANDS + [SINGLE_BLH_BAND])
        .mean()
        .rename([c + "_inst" for c in ["u10", "v10", "u100", "v100", "blh"]])
    )
    l24_s = (
        single.filterDate(_stamp(_shift(date_str, -1), 8), _stamp(date_str, 8))
        .select(SINGLE_WIND_BANDS)
        .mean()
        .rename(["u10_l24", "v10_l24", "u100_l24", "v100_l24"])
    )
    l48_s = (
        single.filterDate(_stamp(_shift(date_str, -2), 8), _stamp(date_str, 8))
        .select(SINGLE_WIND_BANDS)
        .mean()
        .rename(["u10_l48", "v10_l48", "u100_l48", "v100_l48"])
    )

    # --- ERA5-Land (t2m, d2m, sp; precip sums) ------------------------------
    land = ee.ImageCollection(ERA5_LAND)
    inst_l = (
        land.filterDate(inst_start, inst_end)
        .select(LAND_INST_BANDS)
        .mean()
        .rename(["t2m", "d2m", "sp"])
    )
    precip24 = (
        land.filterDate(_stamp(_shift(date_str, -1), 8), _stamp(date_str, 8))
        .select([LAND_PRECIP_BAND])
        .sum()
        .rename("precip_l24")
    )
    precip48 = (
        land.filterDate(_stamp(_shift(date_str, -2), 8), _stamp(date_str, 8))
        .select([LAND_PRECIP_BAND])
        .sum()
        .rename("precip_l48")
    )

    stack = inst_s.addBands(l24_s).addBands(l48_s).addBands(inst_l).addBands(precip24).addBands(precip48)
    result = stack.reduceRegions(
        collection=cells_fc,
        reducer=ee.Reducer.mean(),
        scale=scale_m,
        tileScale=4,
    )
    feats = result.getInfo()["features"]
    band_to_col = {
        "u10_inst": "wind_u10_inst", "v10_inst": "wind_v10_inst",
        "u100_inst": "wind_u100_inst", "v100_inst": "wind_v100_inst",
        "blh_inst": "blh_m_inst",
        "u10_l24": "wind_u10_l24", "v10_l24": "wind_v10_l24",
        "u100_l24": "wind_u100_l24", "v100_l24": "wind_v100_l24",
        "u10_l48": "wind_u10_l48", "v10_l48": "wind_v10_l48",
        "u100_l48": "wind_u100_l48", "v100_l48": "wind_v100_l48",
        "t2m": "t2m_k_inst", "d2m": "d2m_k_inst", "sp": "sp_pa_inst",
        "precip_l24": "precip_l24_m", "precip_l48": "precip_l48_m",
    }
    rows = []
    for f in feats:
        p = f["properties"]
        row = {"cell_id": p["cell_id"], "date": date_str}
        for band, col in band_to_col.items():
            v = p.get(band)
            row[col] = round(float(v), 6) if v is not None else None
        rows.append(row)
    return rows
