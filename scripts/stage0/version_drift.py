"""Stage 0 (D3, risk 1): version drift.

For every S5P OFFL L3_NO2 image since 2022-06, pull processor/algorithm version
properties and aggregate monthly counts. A step change in the monthly mixture =
processor-version drift = the version-shift test period for Phase 3.

Also writes the property names actually present so D3 storage keys are real.
"""
import json

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common_ee import ee_init, ee, save_report

COLLECTION = "COPERNICUS/S5P/OFFL/L3_NO2"
CANDIDATE_PROPS = [
    "PROCESSOR_VERSION",
    "PROCESSING_BASELINE",
    "algorithm_version",
    "processor_version",
    "PRODUCT_ALGORITHM_VERSION",
]


def main() -> None:
    cfg = ee_init()
    col = (
        ee.ImageCollection(COLLECTION)
        .filterDate(cfg["training"]["window_start"], "2026-10-01")
        .select(["tropospheric_NO2_column_number_density"])  # any single band keeps it cheap
    )
    first = col.first()
    # Server objects never compare to client-side None, so detect the real
    # version property from a materialised dictionary (one getInfo, cheap).
    props_info = first.toDictionary().getInfo() if first else {}
    present_props = [p for p in CANDIDATE_PROPS if p in props_info]
    if not present_props:
        raise SystemExit(
            f"None of {CANDIDATE_PROPS} present. Inspect props_info and update "
            "CANDIDATE_PROPS before storing D3 version metadata."
        )
    best = present_props[0]

    monthly = (
        col.map(lambda img: img.set("month", img.date().format("YYYY-MM")))
        .aggregate_histogram("month")
        .getInfo()
    )
    versions_per_month = {}
    months = sorted(monthly)
    for m in months:
        sub = ee.ImageCollection(COLLECTION).filterDate(m + "-01", month_end(m))
        sub = sub.select(["tropospheric_NO2_column_number_density"])
        counter = sub.aggregate_histogram(best).getInfo()
        versions_per_month[m] = counter

    save_report(
        "g0_version_drift",
        {
            "decision": "D3",
            "risk": 1,
            "candidate_props": CANDIDATE_PROPS,
            "props_present_on_first_image": present_props,
            "chosen_property": best,
            "n_monthly_images": sum(monthly.values()),
            "months": months,
            "versions_per_month": versions_per_month,
            "interpretation": "a single dominant version per month = safe; mixed months = step change to flag for the D3 version plot",
        },
    )


def month_end(m: str) -> str:
    from datetime import date, timedelta

    y, mo = map(int, m.split("-"))
    nxt = date(y, mo, 28) + timedelta(days=10)
    return f"{nxt.year}-{nxt.month:02d}-01"
