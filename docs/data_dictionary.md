# Data Dictionary (draft - Phase 1)

One row per (cell_id, date). Batch export by city and month to Parquet.
All columns carry a `_feature_version` in the manifest; the live path must
produce byte-identical values for the same inputs (parity test, plan §8).

## Identity and provenance
- `cell_id` — `<row>/<col>` scheme from `src/no2map/grid.py`; unique per cell.
- `date` — the overpass day (multiple orbits combine to one row; D2).
- `city` — study-area key (`delhi_ncr`, `bengaluru`, `kolkata`, `hyderabad`).
- `processor_version`, `algorithm_version` — per-image, from D3 report keys.
- `n_orbits` — how many orbits/valid pixels contributed.

## Target
- `no2_col_umol_m2` — TROPOMI-retrieved tropospheric column (µmol/m²), D1.
  Valid-area-weighted mean across orbits; `subpixel_spread` alongside.
  **Never** presented as ground-level NO₂ (risk 18).

## Coverage / status (D2, D6)
- `n_valid_fraction` — fraction of cell area covered by valid pixels.
- `state` — observed / reconstructed / insufficient / system_error / no_overpass.
- `reason_code` — clouds / haze / water / low_coverage (+ source code for errors).

## Weather at overpass (D4, D8) — from ERA5-Land hourly (+BLH from single levels)
- `wind_u_10m`, `wind_v_10m` — at matched overpass hour (UTC-hour-from-longitude match).
- `wind_u_100m`, `wind_v_100m` — 100 m winds (better dispersion proxy).
- `temperature_2m`, `dewpoint_temperature_2m` (→ RH derived), `surface_pressure`,
  `precip_l24_m` / `precip_l48_m` (24/48 h SUMS of the DE-ACCUMULATED hourly
  band `total_precipitation_hourly`; the raw `total_precipitation` band
  accumulates from midnight and resets daily, so summing it across midnight
  over-counts), `boundary_layer_height`.
- Lagged (weather only - not NO₂, so live requests see the same feature set):
  `wind_u/v_10m_lag24h`, `wind_u/v_10m_lag48h`, `wind_mean_24h`, `wind_mean_48h`,
  `blh_overpass`, `ventilation = wind_speed × blh`, `precip_lag24h`.

## Static features (computed ~1 km, summarised per cell; D8)
- `elevation_mean_m`, `elevation_std_m` (SRTM).
- `road_density_km_km2`, `major_road_density_km_km2`, `road_density_warn` (OSM/OSMnx; risk 14).
- `dist_nearest_power_plant_km`, `dist_nearest_industrial_km` (risk 13: near-constant drop).
- `wind_aware_emitter_exposure` (Should - built in Phase 2; cut ladder item 6).
- `water_fraction` (JRC GSW; land/mixed/water via 20%/80% cutoffs; sensitivity-checked).
- Optional (ablation only, keep only if beats seed noise): `population`, `built_up_frac`.

## Ambition notes
- No lat/lon in transfer experiments (risk 11).
- `feature_schema_version` bumps on ANY change; parity test pins it.
