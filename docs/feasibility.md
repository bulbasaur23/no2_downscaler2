# Stage 0 Feasibility Note (Gate G0)

**Fill every row from the probe reports in `data/stage0/`. Do not proceed to Phase 1
while a row is empty or guessed.** Report writing date: ____ 2026.

| # | Check (plan Phase 0) | Source report | Finding | Gate impact |
|---|---|---|---|---|
| 1 | Units of `tropospheric_NO2_column_number_density` (expect µmol/m²) and band list | `g0_no2_month.json` | ______ | loss/gain factor if mol/m² |
| 2 | Image count for one month over NCR + n orbit per day | `g0_no2_month.json` | ______ | daily vs weekly fallback (G0 fallback 1) |
| 3 | QA flags present + chosen thresholds (`qa_cloud_fraction_max`, `qa_cirrus_max`) | `g0_no2_month.json` | ______ | observed-cell rule D2 inputs |
| 4 | Valid-pixel fraction after QA masking | `g0_no2_month.json` | ______ | **if too low for daily cells -> weekly composites (fallback)** |
| 5 | What the collection already filters (NRTI absent? QA filter baked in?) | `g0_no2_month.json` | ______ | Phase 1 quality filter scope |
| 6 | Overpass local hour per cell longitude (D4 timezone rule + unit test) | `g0_no2_month.json` | ______ | fixes `weather.overpass_local_hour` config |
| 7 | Version property that EXISTS on images (D3) | `g0_version_drift.json` | ______ | storage keys for Parquet metadata |
| 8 | Version plot: step changes? consistent training window start | `g0_version_drift.json` | ______ | fixes `training.window_start`, defines Nov-2025 shift test period |
| 9 | BLH present in ERA5-Land? (plan expects NO; confirm) | `g0_weather.json` | ______ | risk 21; fallback = drop BLH, keep lagged wind |
| 10 | BLH from ERA5 single levels: band exists + fetch seconds | `g0_weather.json` | ______ | risk 21 timing; cache BLH if slow |
| 11 | Latest available date per weather source (D4 date cap) | `g0_weather.json` | ______ | fixes `weather` date cap + `training.window_end` |
| 12 | Live-request timing (point/region weather) | `g0_weather.json` | ______ | app responsiveness budget |
| 13 | NCR bbox actual cell count (+ all other study areas) | `g0_cells_water.json` | ______ | risk 8 spatial-block feasibility |
| 14 | Water fraction distribution in Delhi NCR; cutoffs 20/80 still sane | `g0_cells_water.json` | ______ | sensitivity basis for Phase 2 |
| 15 | Compute cost of live request + training export (D5) | `g0_quota_cost.json` | ______ | fixes `quota.daily_requests_limit`, app polygon size cap |
| 16 | CPCB access method (manual? API?) -> keep or drop stations | (manual) | ______ | D1 station role: correlation/bias only; drop if manual |
| 17 | Emitter lists: public power-plant list + industrial list, counts per city | (manual) | ______ | risk 13: flag near-constant features |
| 18 | Demo choice: local vs hosted (D7) - decided by #15 | `g0_quota_cost.json` | ______ | Phase 4 shape |
| 19 | Grid + cell-ID scheme written and tested (this repo) | `src/no2map/grid.py` | done (tests) | Plan D2/D8 artifacts |

## G0 decision

- [ ] PASS: proceed to Phase 1 (Delhi pipeline) as planned.
- [ ] PASS with fallback: daily -> weekly composite (row 4), and/or shrink training export (row 15).
- [ ] FAIL: re-scope per plan Gate G0 fallbacks before writing any model code.

## Non-negotiables already fixed in code

- Target = TROPOMI-retrieved column only (D1). Recorded in README, UI, schema.
- 0.05° grid locked (D8) - `config.py` refuses any other `cell_deg`.
- Cell states taxonomy (D6) - `src/no2map/states.py`; `system_error` never cached as data.
- Cell-ID scheme + observed-cell rule (D2) - `src/no2map/grid.py` with tests.
