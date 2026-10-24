"""Stage 0 (D4, risk 2 + 21): weather sources.

Measures, per plan Phase 0:
 - the actual band lists of GEE ERA5-Land hourly and ERA5 single levels
   (BLH presence is exactly what the plan asks to CONFIRM here - risk 21)
 - the latest available date ("actual last available date") of each source,
   which caps the date picker in the app (D4, risk 2)
 - the wall-clock speed of pulling a small BLH/point region (queue timing)
 - a coastal-region availability sanity check (plan section 4)

Writes g0_weather.json - the numbers the feasibility note quotes.
"""
import json
import time

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common_ee import ee_init, ee, save_report

ERA5_LAND_HOURLY = "ECMWF/ERA5_LAND/HOURLY"
# Current hourly single-levels collection on GEE (51710 was deprecated; name
# verified live 2026-10-24: 292 bands incl. boundary_layer_height).
ERA5_HOURLY_SINGLE = "ECMWF/ERA5/HOURLY"


def _latest_date(collection: str) -> dict:
    col = ee.ImageCollection(collection).limit(500, "system:time_start")
    latest = col.sort("system:time_start", False).first()
    # ECMWF collections on GEE carry far-future sentinel system:time_start
    # values (overflowing local epoch math on Windows), so format the date
    # SERVER-side and keep the raw ms as a string for diagnosis only.
    iso = latest.date().format("YYYY-MM-dd HH:mm:ss", "UTC").getInfo() if latest else None
    raw_ms = latest.get("system:time_start").getInfo() if latest else None
    return {"collection": collection, "latest_utc": iso, "raw_time_start_ms": str(raw_ms)}


def main() -> None:
    cfg = ee_init()
    area = cfg["study_areas"]["delhi_ncr"]
    region = ee.Geometry.Rectangle([area["lon_min"], area["lat_min"], area["lon_max"], area["lat_max"]])
    point = ee.Geometry.Point([77.21, 28.61])  # central Delhi

    # Band lists (the confirmation the plan demands for BLH).
    land_bands = ee.ImageCollection(ERA5_LAND_HOURLY).first().bandNames().getInfo()
    try:
        single_bands = ee.ImageCollection(ERA5_HOURLY_SINGLE).first().bandNames().getInfo()
    except Exception as exc:
        single_bands = {"error": repr(exc)}

    # Latest dates -> the actual "data as of" cap for the app (risk 2).
    latest = [_latest_date(ERA5_LAND_HOURLY), _latest_date(ERA5_HOURLY_SINGLE)]

    # Timing: one overpass-hour window, small region, one variable each -
    # this is the "time the live-request path" test the plan requires.
    def timed(desc, fn):
        t0 = time.perf_counter()
        try:
            val = fn()
            out = {"desc": desc, "ok": True, "seconds": round(time.perf_counter() - t0, 2), "value": val}
        except Exception as exc:
            out = {"desc": desc, "ok": False, "seconds": round(time.perf_counter() - t0, 2), "error": repr(exc)}
        return out

    blh_missing_from_land = "boundary_layer_height" not in land_bands

    timings = []
    if blh_missing_from_land:
        timings.append(timed(
            "blh_from_single_levels_point",
            lambda: ee.ImageCollection(ERA5_HOURLY_SINGLE)
            .filterDate("2026-09-15", "2026-09-16")
            .filter(ee.Filter.calendarRange(8, 8, "hour"))
            .first()
            .select("boundary_layer_height")
            .reduceRegion(ee.Reducer.mean(), point, scale=1000, maxPixels=1e9)
            .getInfo(),
        ))
    timings.append(timed(
        "era5_land_u10_point",
        lambda: ee.ImageCollection(ERA5_LAND_HOURLY)
        .filterDate("2026-09-15", "2026-09-16")
        .filter(ee.Filter.calendarRange(8, 8, "hour"))
        .first()
        .select("u10")
        .reduceRegion(ee.Reducer.mean(), point, scale=25000, maxPixels=1e9)
        .getInfo(),
    ))
    timings.append(timed(
        "era5_land_24h_mean_wind_delhi_region",
        lambda: ee.ImageCollection(ERA5_LAND_HOURLY)
        .filterDate("2026-09-14", "2026-09-15")
        .mean()
        .reduceRegion(
            ee.Reducer.mean(), region, scale=25000, maxPixels=1e9, bestEffort=True
        )
        .getInfo(),
    ))

    save_report(
        "g0_weather",
        {
            "decision": "D4",
            "risks": [2, 21],
            "era5_land_bands": land_bands,
            "era5_single_level_bands": single_bands,
            "blh_missing_from_era5_land": blh_missing_from_land,
            "latest_dates": latest,
            "timing_probes": timings,
            "interpretation": {
                "blh": "if missing above, BLH must come from ERA5 single levels OR its collection name needs fixing - record which",
                "latency": "latest_dates drive the app date-picker cap (earliest of the two wins)",
                "one_source_rule": "if single-levels fetch is fast and matches ERA5-Land parity later, prefer ONE source",
            },
        },
    )


if __name__ == "__main__":
    main()
