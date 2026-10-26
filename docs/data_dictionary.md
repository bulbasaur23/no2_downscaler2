# Data Dictionary (Phase 1, updated to the shipped schema)

One row per `(cell_id, date)`. Exported by city and month to Parquet; every
month is pinned by the manifest entry that produced it.

`feature_schema_version`, `processor_versions` and the export path live in
`data/exports/manifest.json` (per city-month), not as per-row columns: a row
is only valid together with the manifest entry that says which schema and
source versions produced it. The live path refuses a schema mismatch.

## Identity and provenance
- `city` — study-area key (`delhi_ncr`, `bengaluru`, `kolkata`, `hyderabad`).
- `cell_id` — `r<row:05d>c<col:06d>` in global 0.05° grid space
  (e.g. `r02056c005146`); inverse is `no2map.grid.parse_cell_id`.
- `date` — overpass day, `YYYY-MM-DD` (multiple orbits combine into one row; D2).

## Target (D1)
- `value_umol_m2` — TROPOMI L3 tropospheric NO₂ column in **µmol/m²**,
  valid-area-weighted mean across the day's orbits. **Never** ground-level NO₂.
- `n_valid_fraction` — fraction of the cell covered by QA-passed pixels.
- `subpixel_spread_umol_m2` — within-cell spread of valid pixel values.
- `n_orbit_overlap` — mean number of orbits contributing per valid pixel.

## Coverage / status (D2, D6)
- `state` — `observed` / `insufficient` / `no_overpass` in the training panel;
  the live path adds `reconstructed` (Model A fills non-observed cells) and
  `unavailable` (the fetch returned nothing for that cell).
- `reason_code` — `low_coverage`, `no_overpass`, `fetch_missing`, or a source
  code (`gee`, `osm`, `weather`, `outage`) for `system_error`.

## Weather at overpass (D4, D8) — ERA5-Land hourly (+ BLH from ERA5 single levels)
- `t2m_k_inst`, `d2m_k_inst` — 2 m temperature / dewpoint (K) at the matched
  overpass hour (UTC hour from cell longitude).
- `sp_pa_inst` — surface pressure (Pa).
- `wind_u10_inst`, `wind_v10_inst` — 10 m wind components (m/s).
- `wind_u100_inst`, `wind_v100_inst` — 100 m wind components (m/s).
- `blh_m_inst` — boundary-layer height at the overpass hour (m).
- `precip_l24_m`, `precip_l48_m` — 24/48 h precipitation sums (m), built from
  the de-accumulated hourly band (the raw `total_precipitation` band
  accumulates from midnight, so summing it across midnight over-counts).
- Lagged (weather only, so live requests see the same inputs):
  `wind_u10_l24`, `wind_v10_l24`, `wind_u100_l24`, `wind_v100_l24`,
  `wind_u10_l48`, `wind_v10_l48`, `wind_u100_l48`, `wind_v100_l48`.

## Static features (computed once per city at ~1 km, summarised per cell; D8)
- `elevation_mean_m`, `elevation_std_m`, `elevation_min_m`, `elevation_max_m`
  (SRTM 90 m).
- `water_fraction` — ESA WorldCover v200 class 80 (permanent water), mean per
  cell.
- `water_state` — `land` / `mixed` / `water` from the configured cutoffs.

## Derived at feature-build time (`no2map.features.build_features`)
Never stored as raw inputs; both training and live derive them through the
same function (parity contract):
`t2m_deg_c`, `rh_pct`, `sp_hpa_inst`, `wind_speed_inst`,
`wind_dir_cos_inst`, `wind_dir_sin_inst`, `ventilation_proxy`,
`blh_x_precip_proxy`, `wind_speed_l24`, `ventilation_l24`, `wind_speed_l48`,
`ventilation_l48`, `month_sin`, `month_cos`.

## Known gaps vs the v3 plan (stated, not hidden)
- **Roads and emitters are not in the schema.** `data_dictionary.md` used to
  list `road_density_km_km2`, `major_road_density_km_km2`,
  `road_density_warn`, `dist_nearest_power_plant_km`,
  `dist_nearest_industrial_km` and `wind_aware_emitter_exposure`; none of them
  are computed. Phase 2 shipped elevation + water only.
- **The emitter/road data sources (OSM/osmnx, power-plant list) are not
  wired.** No per-city emitter counts exist yet, so the Phase-5 check
  "emitter counts per city" cannot be run.
- **MLflow tracking is not wired** (Complete-tier item).
- Population / built-up ablations are not built (Stretch-tier item).
- No lat/lon features anywhere (risk 11) — transfer experiments stay
  coordinate-free, and the OOD flag uses static features only.

## Conventions
- `feature_schema_version` bumps on ANY feature change; the parity test and
  the serving artifact both pin it. The panel loader refuses a month whose
  manifest schema differs from the installed code (and refuses a mix of
  schemas across months), so training can never silently consume columns the
  serving path does not produce — re-export the city instead.
- Target-less rows (`insufficient`, `no_overpass`) carry no value and are
  never fitted; they are kept in the panel so a missing day cannot be confused
  with a day the worker never ran.
