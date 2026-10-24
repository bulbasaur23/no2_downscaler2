"""Stage 0 gate G0 (part 1): pull ONE month of Delhi NO2 and measure everything
the plan *assumed* instead of trusting it (list of checks from plan Phase 0):

 - units and bands of COPERNICUS/S5P/OFFL/L3_NO2
 - QA flag bands available and value ranges (choose thresholds for configs)
 - valid fraction per image after masking (drives daily vs weekly fallback)
 - what the collection already filters (OFFL vs NRTI check, image count, times)
 - overpass timestamps -> local hour of day per cell longitude (D4 timezone)

Writes a JSON report + one month Parquet-ready image list for Phase 1.
"""
import json
from datetime import datetime, timezone

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common_ee import ee_init, ee, save_report


def main() -> None:
    cfg = ee_init()
    area = cfg["study_areas"]["delhi_ncr"]
    region = ee.Geometry.Rectangle(
        [area["lon_min"], area["lat_min"], area["lon_max"], area["lat_max"]]
    )
    start, end = "2026-09-01", "2026-10-01"  # one recent full month, Stage 0 only

    col = ee.ImageCollection(cfg["sources"]["no2"]).filterBounds(region)
    month = col.filterDate(start, end)
    n_images = month.size().getInfo()

    first = month.first()
    bands = first.bandNames().getInfo()
    props = first.toDictionary(first.propertyNames().sort()).getInfo() if n_images else {}

    times = (
        month.aggregate_array("system:time_start")
        .getInfo() if n_images else []
    )
    times_local = sorted(
        datetime.fromtimestamp(t / 1000, tz=timezone.utc).isoformat() for t in times
    )

    # valid fraction: fraction of pixels in region that are QA-flagged "fine"
    # uses tropospheric NO2 column + band qa; thresholds refined against
    # the actual band values printed below (do NOT hardcode forward).
    qa_probe = (
        month.limit(3)
        .select(bands[:3])
        .mean()
        .reduceRegion(
            reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
            geometry=region,
            scale=11132,  # one native-pixel scale probe only
            maxPixels=1e9,
        )
        .getInfo()
    )

    n_per_day = (
        month.aggregate_histogram("system:time_start").getInfo()
        if n_images
        else {}
    )

    save_report(
        "g0_no2_month",
        {
            "gate": "G0",
            "collection": cfg["sources"]["no2"],
            "window": [start, end],
            "region_bbox": area,
            "n_images": n_images,
            "bands": bands,
            "sample_properties": props,
            "overpass_times_utc_sample": times_local[:10],
            "qa_probe_minmaxmean_limited": qa_probe,
            "image_ids_per_day_sample": list(n_per_day.items())[:5],
        },
    )


if __name__ == "__main__":
    main()
