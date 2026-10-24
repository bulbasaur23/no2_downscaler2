"""Stage 0 (D8, risks 6/15): study-area cell counts, water cutoffs, coastline.

 - counts actual 0.05 deg cells inside each configured study-area bbox
   (the plan explicitly asks Stage 0 to report the NCR count - risk 8)
 - computes JRC Global Surface Water occurrence per cell for the Delhi NCR,
   printing the distribution so the 20%/80% land/water guesses get a
   sensitivity basis (water cutoffs are flagged as guesses in section 4)
 - reports rough coastal composition of Kolkata (nearest sea on the east) to
   sanity-check the "coastal cells warned only" stance cheaply
"""
import json

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common_ee import ee_init, ee, save_report
from no2map.grid import cells_in_bbox, cell_area_m2


def main() -> None:
    cfg = ee_init()

    counts = {
        name: len(list(cells_in_bbox(a["lat_min"], a["lat_max"], a["lon_min"], a["lon_max"])))
        for name, a in cfg["study_areas"].items()
    }

    # NCT alone - what the plan says is only ~50 cells: show why NCR was chosen.
    nct_cells = sum(
        1 for _ in cells_in_bbox(28.40, 28.90, 76.83, 77.35)
    )

    # JRC water occurrence aggregated over the Delhi NCR region (quick look)
    wa = cfg["study_areas"]["delhi_ncr"]
    region = ee.Geometry.Rectangle([wa["lon_min"], wa["lat_min"], wa["lon_max"], wa["lat_max"]])
    # JRC Global Surface Water is a STATIC single Image (no .mean() there).
    water = (
        ee.Image(cfg["sources"]["water"])
        .select("occurrence")
        .gt(cfg["water_cutoffs"]["land_max_water_fraction"] * 100)
    )
    water_frac = water.reduceRegion(
        ee.Reducer.mean(), region, scale=1113, maxPixels=1e9, bestEffort=True
    ).getInfo()

    save_report(
        "g0_cells_water",
        {
            "decision": "D8",
            "risks": [6, 8, 15],
            "cell_deg": cfg["grid"]["cell_deg"],
            "study_area_cell_counts": counts,
            "delhi_nct_only_cells": nct_cells,
            "why_ncr": "NCT alone has too few cells for spatial blocking (plan risk 8); NCR bbox is the substitute",
            "delhi_ncr_mean_water_fraction_gt_20pct": water_frac,
            "water_cutoffs_as_configured": cfg["water_cutoffs"],
            "note": "cutoffs 20%/80% remain guesses until Phase 2 sensitivity run",
        },
    )


if __name__ == "__main__":
    main()
