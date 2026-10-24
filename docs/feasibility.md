# Stage 0 Feasibility Note (Gate G0)

**Measured 2026-10-24** from the probe reports in `data/stage0/` under GEE project
`alan-no2-downscaler` (noncommercial tier). **Verdict at the bottom.**

| # | Check (plan Phase 0) | Source report | Finding | Gate impact |
|---|---|---|---|---|
| 1 | Units of tropospheric NO₂ column | `g0_no2_month.json`, `g0_quota_cost.json` | Bands are **mol/m²** (monthly NCR mean 3.7–4.5e-05 mol/m² ≈ 37–45 µmol/m²) | Phase 1 multiplies by **1e6** to store µmol/m² (D1) |
| 2 | Images per month + orbits/day | `g0_no2_month.json` | **426 images/30 days (~14/day)**; exactly **ONE orbit/day covers Delhi at UTC 08:00** (= 13:30 IST) on the sample day | Daily cell value = single orbit weighted mean; multi-orbit weighting (D2) still implemented for overlap days |
| 3 | QA flags + thresholds | `g0_no2_month.json` | L3 has **no qa_value band**; GEE applies server-side QA masking already; `cloud_fraction` band present (month mean 0.26, max 0.36) | `observed_cell_rule.qa_*` config entries are moot for L3 - Phase 1 keeps only the valid-pixel regridding rule (D2) |
| 4 | Valid-pixel fraction for daily cells | `g0_no2_month.json` | Collection returns valid data over Delhi at the overpass hour; per-cell valid fraction measured in Phase 1 from unmasked counts | Fallback (weekly composites) stays armed, not triggered |
| 5 | What the collection already filters | `g0_no2_month.json` | OFFL L3: QA pre-masked server-side (qa<0.75 for tropospheric); OFFL not NRTI confirmed | Phase 1 composites orbits with valid-area weighting only |
| 6 | Overpass local hour (D4 timezone rule) | `g0_no2_month.json` | Delhi observed at **UTC hour 8 = 13:30 IST** on sample day | `weather.overpass_local_hour: 13.5` confirmed |
| 7 | Version property that exists (D3) | `g0_version_drift.json`, `g0_no2_month.json` | **`PROCESSOR_VERSION`** (e.g. `02.09.01`) + `ALGORITHM_VERSION` (1.8.0) present; old-format ids (`2.4.0`…) and new (`02.09.01`) coexist historically | D3 metadata keys fixed |
| 8 | Version plot: step changes, training window | `g0_version_drift.json` | Transitions: 2022-07→`2.4.0`, 2023-03→`2.5.0`, 2023-11→`2.6.0`, 2024-09→`2.7.1`, 2024-11→`2.8.0`, **2025-11→`02.09.01`** (the plan's Nov-2025 update, confirmed). Every transition has a mixed month. "Mid-2022 single version" does NOT hold | Primary window **2023-04-01..2025-09-30 excluding transition months {2024-09, 2024-11}** (incl. 2 winters); era stored per month in the manifest; era-stratified scoring + version-shift test on 2025-11+ defined. Final tuning only at Gate G3 |
| 9 | BLH in ERA5-Land? (risk 21) | `g0_weather.json` | **Confirmed missing** (`blh_missing_from_era5_land: true`) | BLH from ERA5/HOURLY, as planned |
| 10 | BLH from ERA5 single levels: exists + speed | `g0_weather.json` | Band present in `ECMWF/ERA5/HOURLY`; point fetch **0.6–0.74 s** | Fast enough for live (cached anyway); risk 21 fallback NOT needed |
| 11 | Latest available date (D4 cap) | `g0_weather.json` | ERA5-Land latest **2026-10-18 23:00 UTC**; ERA5/HOURLY latest **2026-10-18 14:00 UTC** (~6 days behind) | App date picker cap = **2026-10-18** (moving); better than the plan's 2–3 month assumption |
| 12 | Live weather timing | `g0_weather.json` | BLH point 0.60 s; ERA5-Land 24 h region mean 1.72 s. (One probe failed on a legacy band name - config fixed, path re-verified in Phase 1) | Live request budget comfortably <5 s |
| 13 | Actual cell counts (D8, risk 8) | `g0_cells_water.json` | **delhi_ncr 1610**, delhi NCT-bbox-only 132, bengaluru 156, kolkata 169, hyderabad 120 | NCR bbox decision validated (13× the NCT-only count); spatial blocking feasible |
| 14 | Water fraction + cutoffs (risk 15) | `g0_cells_water.json` | **JRC GSW occurrence raster broken on GEE** (see anomaly A2); switched to ESA WorldCover v200 class 80: NCR mean permanent-water **0.53%** ✓ plausible; point classes sane (50=Built-up) | Water source = WorldCover; 20%/80% cutoffs remain a Phase-2 sensitivity item |
| 15 | Compute cost (D5) | `g0_quota_cost.json` | Live reduce 1.4–1.5 s; monthly training reduce 0.67–0.92 s | Daily request limit set after Phase-1 export measurement; polygon size cap from Phase 4 cache/timing |
| 16 | CPCB access | (manual) | Not confirmed - keep as optional per plan (correlation/bias only) | Dropped unless access simplifies; does not block |
| 17 | Emitter lists | (manual) | Pending Phase 2; flag near-constant features per risk 13 | Does not block Phase 1 |
| 18 | Demo choice (D7) | `g0_quota_cost.json` | Timing OK for local live; hosted endpoint deferred to Stretch | Local + cached demo mode |
| 19 | Grid + cell-ID scheme + observed rule | `src/no2map/grid.py` | Written with 22 tests (integer millidegree bucketing; boundary coordinates snap to the higher cell by convention) | Single source of truth for cells everywhere |

## Anomalies found (recorded verbatim - these were assumptions killed by measurement)

- **A1 `filterBounds` is a no-op filter re footprints** on `OFFL/L3_*`: each image's
  `system:footprint` is the full globe, so selection must use the overpass UTC-hour
  window (Delhi = 08:00). Implemented as a Phase-1 requirement, tested by parity.
- **A2 JRC/GSW1_4/GlobalSurfaceWater occurrence raster mostly masked on GEE:**
  point probes return null worldwide; a 0.05° cell over Delhi exposes 46 unmasked
  pixels where ~34,000 exist; FIXED-histogram bin counts inconsistent with the
  geometry. Water source switched to `ESA/WorldCover/v200/2021` (class 80), which
  the plan itself listed as the alternative (section 4).
- **A3 ECMWF collections carry far-future sentinel timestamps AND a deep
  pre-1970 backfill:** naive `limit().sort()` picks the wrong tail; local epoch
  math overflows Windows. All date logic must filterDate a window first and
  format dates server-side (`ee.Date().format`).
- **A4 ERA5-Land variable names differ from plan shorthand** (`u10` is
  `u_component_of_wind_10m` etc.), and ERA5-Land has NO 100 m wind and NO BLH.
  Config updated to real band names; weather source parity test (D4) still due in
  Phase 1 to decide: all-ERA5/HOURLY single source vs ERA5-Land+BLH mix.

## G0 decision

- [x] **PASS** - proceed to Phase 1 (Delhi pipeline) with the three recorded
      adjustments: (a) mol/m² → µmol/m² conversion, (b) overpass-hour image
      selection instead of footprint filtering, (c) training window
      2023-04..2025-09 minus transition months, era per month in manifest.
- Fallbacks armed but NOT triggered: weekly composites (row 4), shrunken export (row 15).
