"""Stage 0 (D5, risk 3): GEE quota cost measurement.

Measures the compute cost of the two request shapes the platform makes:
 1. one live request: mean over a Delhi-sized polygon for one image,
 2. one training export: one month over the full NCR bbox.

ee.Image.reduceRegion(...).getInfo() surfaces a "numCPU" style usage signal
only on the actual ExecutionInfo; GEE does not return cost directly, so we
measure wall time and record per-tier behaviour (the non-commercial quota is
time-based). Numbers here set configs.quota.daily_requests_limit.
"""
import json
import time

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common_ee import ee_init, ee, save_report


def timed(desc, fn):
    t0 = time.perf_counter()
    try:
        val = fn()
        return {"desc": desc, "ok": True, "seconds": round(time.perf_counter() - t0, 2), "value": val}
    except Exception as exc:
        return {"desc": desc, "ok": False, "seconds": round(time.perf_counter() - t0, 2), "error": repr(exc)}


def main() -> None:
    cfg = ee_init()
    area = cfg["study_areas"]["delhi_ncr"]
    region = ee.Geometry.Rectangle([area["lon_min"], area["lat_min"], area["lon_max"], area["lat_max"]])

    col = ee.ImageCollection(cfg["sources"]["no2"]).filterDate("2026-09-15", "2026-09-16")

    probes = [
        timed(
            "live_request_one_polygon_one_image",
            lambda: col.first()
            .select("tropospheric_NO2_column_number_density")
            .reduceRegion(ee.Reducer.mean(), region, scale=1113, maxPixels=1e9)
            .getInfo(),
        ),
        timed(
            "live_request_small_polygon",
            lambda: col.first()
            .select("tropospheric_NO2_column_number_density")
            .reduceRegion(
                ee.Reducer.mean(),
                ee.Geometry.Rectangle([77.0, 28.4, 77.4, 28.8]),
                scale=1113,
                maxPixels=1e9,
            )
            .getInfo(),
        ),
        timed(
            "training_export_one_month_reduce_bestEffort",
            lambda: col.mean()
            .select("tropospheric_NO2_column_number_density")
            .reduceRegion(
                ee.Reducer.mean(), region, scale=1113, maxPixels=int(1e9), bestEffort=True
            )
            .getInfo(),
        ),
    ]

    save_report(
        "g0_quota_cost",
        {
            "decision": "D5",
            "risk": 3,
            "tier_configured": cfg["quota"]["tier"],
            "project_configured": cfg["quota"]["project_id"],
            "probes": probes,
            "interpretation": {
                "rule": "wall times feed manual quota-log budgeting for the noncommercial tier; keep ONE project (D5)",
                "next": "if small-polygon cost is high, tighten the polygon size cap before Phase 4",
                "cache": "each result cell must cache first; demo mode stays the outage fallback (D7)",
            },
        },
    )


if __name__ == "__main__":
    main()
