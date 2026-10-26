"""NO2 acquisition (Phase 1 / Gate G1): orbit compositing to 0.05-deg cells.

Concept: one row per (cell, date). The measured Stage-0 facts this module
encodes (docs/feasibility.md):
 - OFFL L3 footprints are global -> images must be selected by the overpass
   UTC-hour window of the study area, NOT by filterBounds (anomaly A1),
 - QA is already applied server-side -> composite valid pixels only,
 - Delhi is observed at UTC hour 8 (13:30 IST); per-city windows derive from
   S5P's ~13:30 local-solar overpass: h0 = 13.5 - mean_lon/15 (D4 timezone rule),
 - L3 band values are mol/m^2 -> multiplied by 1e6 to store umol/m^2 (D1).

The D2 observed-cell arithmetic lives in pure helpers (testable without GEE)
and is applied to per-cell pixel sums returned from Earth Engine.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import ee

MOL_TO_UMOL = 1e6  # measured: GEE OFFL/L3_NO2 bands are mol/m^2 (D1 target stored umol/m^2)
TARGET_BAND = "tropospheric_NO2_column_number_density"


# --------------------------------------------------------------------------
# Pure, testable helpers (no GEE calls)
# --------------------------------------------------------------------------


def overpass_utc_hours(lon_min: float, lon_max: float) -> list[int]:
    """UTC hours whose orbits can cover this longitude range at ~13:30 local solar.

    S5P overpasses near local solar 13:30; local solar hour over longitude L at
    UTC hour h is h + L/15. Inverting: h0 = 13.5 - mean_lon/15.

    The window is widened to +/- 1 hour bucket beyond the two envelope buckets
    of h0: orbits are spaced ~101 min, so the day's Delhi-covering orbit drifts
    its time_start hour by about one bucket day-to-day (measured in Phase 1:
    a [8,9] window caught only 6 of 31 January days; the rest were no_overpass).
    Orbits centered far away simply contribute no pixels - and if two orbits DO
    see a pixel, D2's valid-area weighting merges them - so wider is safe."""
    h0 = 13.5 - (lon_min + lon_max) / 2.0 / 15.0
    lo = int(math.floor(h0)) - 1
    hi = int(math.ceil(h0)) + 1
    return list(range(lo, hi + 1))


def month_days(month: str) -> list[str]:
    """All dates of a 'YYYY-MM' month as 'YYYY-MM-DD'."""
    year, mon = map(int, month.split("-"))
    d = datetime(year, mon, 1)
    out = []
    while d.month == mon:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def cell_stats_from_sums(value_sum: float, value_sq_sum: float, valid_sum: float,
                         one_sum: float, orbit_count_sum: float,
                         main_threshold: float = 0.5) -> dict:
    """D2 arithmetic from reduceRegions SUM statistics of a day's composite.

    value / value_sq: pixel-level sums feeding the valid-area-weighted cell
    value and the within-cell ("sub-pixel") spread; valid: 1 where the pixel is
    present (QA-passed), 0 where masked; one: constant 1 -> one_sum counts
    every pixel center inside the cell so totals are grid-defined rather than
    reduction-defined. Applies the D2 threshold to decide OBSERVED.
    """
    if one_sum <= 0:
        raise ValueError("one_sum must be > 0 (no pixel centers fell inside the cell)")
    total = float(one_sum)
    valid_f = float(valid_sum) / total

    if float(valid_sum) <= 0:
        return {
            "value_umol_m2": None,
            "n_valid_fraction": 0.0,
            "subpixel_spread_umol_m2": None,
            "n_orbit_overlap": round(float(orbit_count_sum) / max(total, 1.0), 3),
            "state": "no_overpass",  # D6: no orbit data -> its own state, not insufficient
            "reason_code": "no_overpass",
        }

    mean = float(value_sum) / float(valid_sum)
    var = float(value_sq_sum) / float(valid_sum) - mean * mean
    spread = math.sqrt(max(var, 0.0))  # population estimate from sums
    observed = valid_f >= main_threshold
    return {
        "value_umol_m2": round(mean * MOL_TO_UMOL, 4),
        "n_valid_fraction": round(valid_f, 4),
        "subpixel_spread_umol_m2": round(spread * MOL_TO_UMOL, 4),
        "n_orbit_overlap": round(float(orbit_count_sum) / max(float(valid_sum), 1.0), 3),
        "state": "observed" if observed else "insufficient",
        "reason_code": None if observed else "low_coverage",
    }


# --------------------------------------------------------------------------
# Earth Engine side
# --------------------------------------------------------------------------


def overpass_hour_filter(hours: list[int]) -> ee.Filter:
    """ee.Filter matching the given UTC calendar hours."""
    parts = [ee.Filter.calendarRange(h, h, "hour") for h in hours]
    return parts[0] if len(parts) == 1 else ee.Filter.Or(*parts)


def cells_featurecollection(city_cfg: dict) -> ee.FeatureCollection:
    """0.05-deg cells of the study bbox as GEE polygons carrying cell_id."""
    from no2map.grid import cells_in_bbox

    feats = []
    for c in cells_in_bbox(city_cfg["lat_min"], city_cfg["lat_max"], city_cfg["lon_min"], city_cfg["lon_max"]):
        geom = ee.Geometry.Rectangle([c.lon_min, c.lat_min, c.lon_max, c.lat_max])
        feats.append(ee.Feature(geom, {"cell_id": c.id}))
    return ee.FeatureCollection(feats)


def no2_projection_scale() -> float:
    """Native scale (in metres) of the L3 target band, measured once per run.

    D2 says 'check the native scale of the GEE band before regridding' - this
    is that check, recorded in the log where the manifest can see it."""
    img = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2").first().select(TARGET_BAND)
    return img.projection().nominalScale().getInfo()


def day_no2_cell_sums(city_cfg: dict, date_str: str, cells_fc: ee.FeatureCollection,
                      scale_m: float) -> list[dict]:
    """Per-cell SUM statistics for one date's orbit composite.

    Server-side (single getInfo): filter the OFFL L3 collection to the date and
    the city's overpass UTC-hour window; mean/second-moment across contributing
    orbits per pixel; add orbit-overlap count, a valid-indicator band, and a
    constant-1 band for grid-defined totals; reduceRegions over all cells with
    a SUM reducer at the L3 native scale.

    Rows carry exactly the inputs of cell_stats_from_sums plus cell_id; the D2
    thresholding and unit conversion happen client-side (pure, tested).

    Returns [] when the date has no image in the overpass window: a date with
    no satellite pass is a D6 fact, not an error (callers turn it into
    `no_overpass` rows via `day_rows`)."""
    year, mon, day = map(int, date_str.split("-"))
    start = ee.Date(f"{year}-{mon:02d}-{day:02d}")
    hours = overpass_utc_hours(city_cfg["lon_min"], city_cfg["lon_max"])

    ic_day = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
        .filterDate(start, start.advance(1, "day"))
        .filter(overpass_hour_filter(hours))
        .select(TARGET_BAND)
    )

    if int(ic_day.size().getInfo()) == 0:
        # Nothing matched this date + overpass window. Building a composite
        # from an empty collection gives a BANDLESS image, and `.pow(2)` then
        # fails server-side ("Image.pow: ... Got 0 and 1"). A date with no
        # overpass is a data fact (D6 `no_overpass`), not a fetch failure, so
        # signal it to the caller instead of crashing the whole month.
        return []

    comp = ic_day.mean().rename("value")  # valid-only weighted mean across orbits
    stack = (
        comp.addBands(comp.pow(2).rename("value_sq"))
        .addBands(comp.unmask(-1).neq(-1).rename("valid"))  # 1 where composite has data
        .addBands(ee.Image.constant(1).rename("one"))       # grid-defined pixel total
        .addBands(ic_day.count().rename("orbit_count"))     # how many orbits contributed
    )
    result = stack.reduceRegions(
        collection=cells_fc,
        reducer=ee.Reducer.sum(),
        scale=scale_m,
        tileScale=4,
    )
    features = result.getInfo()["features"]
    rows = []
    for f in features:
        p = f["properties"]
        rows.append(
            {
                "cell_id": p["cell_id"],
                "value_sum": p.get("value") or 0.0,
                "value_sq_sum": p.get("value_sq") or 0.0,
                "valid_sum": float(p.get("valid") or 0.0),
                "one_sum": float(p.get("one") or 0.0),
                "orbit_count_sum": float(p.get("orbit_count") or 0.0),
            }
        )
    return rows


def month_processor_versions(year: int, mon: int) -> list[str]:
    """Distinct PROCESSOR_VERSION values used by the month's images (D3)."""
    start = f"{year}-{mon:02d}-01"
    if mon == 12:
        end = f"{year + 1}-01-01"
    else:
        end = f"{year}-{mon + 1:02d}-01"
    ic = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2").filterDate(start, end)
    vals = ic.aggregate_array("PROCESSOR_VERSION").distinct().getInfo()
    return sorted(str(v) for v in vals)


def day_processor_versions(date_str: str, hours: list[int] | None = None) -> list[str]:
    """PROCESSOR_VERSION values of the images that could cover the city on a date.

    A processor update does not land on a month boundary: measured on Delhi's
    overpass window, 2025-11 was `2.8.0` until the 21st, mixed on the 22nd and
    `02.09.01` from the 23rd. A month-level version list (over all global
    orbits) therefore cannot answer "is this eval day under the new processor?"
    - this per-day, overpass-windowed call can. Used by the version-shift
    runner to keep only version-pure eval days; it is not a data value."""
    start = ee.Date(date_str)
    ic = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
          .filterDate(start, start.advance(1, "day")))
    if hours:
        ic = ic.filter(overpass_hour_filter(hours))
    vals = ic.aggregate_array("PROCESSOR_VERSION").distinct().getInfo()
    return sorted(str(v) for v in (vals or []))


def convert_day_rows(sums_rows: list[dict], main_threshold: float) -> list[dict]:
    """Apply the pure D2 arithmetic to each cell's sums of one day."""
    out = []
    for r in sums_rows:
        stats = cell_stats_from_sums(
            r["value_sum"], r["value_sq_sum"], r["valid_sum"],
            r["one_sum"], r["orbit_count_sum"], main_threshold=main_threshold,
        )
        out.append({"cell_id": r["cell_id"], **stats})
    return out


def no_overpass_rows(cell_ids: list[str]) -> list[dict]:
    """D6 `no_overpass` row for every requested cell on a date with no imagery.

    A day whose overpass window matched zero images has no cell values at all -
    but the (cell, date) panel still needs a row per cell, otherwise the day
    silently disappears from the export and month checks cannot tell
    "no satellite pass" apart from "worker never ran". These rows are a data
    fact (D6), carry no value, and are never fitted or cached as data.
    """
    return [
        {
            "cell_id": cid,
            "value_umol_m2": None,
            "n_valid_fraction": 0.0,
            "subpixel_spread_umol_m2": None,
            "n_orbit_overlap": 0.0,
            "state": "no_overpass",
            "reason_code": "no_overpass",
        }
        for cid in cell_ids
    ]


def day_rows(sums_rows: list[dict], cell_ids: list[str],
             main_threshold: float = 0.5) -> list[dict]:
    """The D6 rows for one date, covering every requested cell.

    Single entry point for both the month exporter and the live path so they
    cannot drift: an empty day (no image in the overpass window) becomes
    `no_overpass` rows rather than an empty list; otherwise the D2 arithmetic
    in `convert_day_rows` is applied as usual.
    """
    if not sums_rows:
        return no_overpass_rows(cell_ids)
    return convert_day_rows(sums_rows, main_threshold)
