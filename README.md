# AI-Based NO₂ Mapping Platform

Mapping of the **TROPOMI-retrieved tropospheric NO₂ column (µmol/m²)** over Indian cities on a
**0.05° grid**, with a live *draw-any-polygon* query app, gap context, calibrated uncertainty,
and **honest validation**.

> **What this is NOT:** this is not ground-truth NO₂. The target is the satellite-retrieved
> product (GEE `COPERNICUS/S5P/OFFL/L3_NO2`). Accuracy is reported **only against TROPOMI**.
> Station data (CPCB, optional) gives correlation and bias only — never accuracy (D1).

## Concept

1. **Fixed 0.05° grid** — TROPOMI's native pixel is ~5.5 × 3.5 km, so a finer grid would repeat
   one retrieval across many cells and inflate the apparent sample size (D8). Static layers
   (roads, elevation, emitters) are sampled at ~1 km and *summarised per cell*.
2. **Draw-any-polygon live feature** — every input the model needs is available at request time
   from data sources with short latency (weather + static layers), so a user can draw any
   polygon over India and get a map without retraining.
3. **Honest validation** — two-level scoring (city-daily level AND within-city spatial pattern),
   spatial blocks sized from residual autocorrelation (variogram), Model A must beat per-cell
   *climatology* or be reported as such, and results are labelled with units, coverage, model
   version, and data versions.

## Scope tiers (from the v3 plan)

| Tier | Contents | Target |
|---|---|---|
| **Must (MVP)** | Delhi pipeline; XGBoost Model A vs baselines under blocked and chronological splits; calibrated intervals; polygon app for any Indian region with data (draw, date, map); cell states observed / insufficient; exports; README with limits | Day 28 |
| **Should (Complete)** | Bengaluru + Kolkata training; leave-one-city-out transfer; Hyderabad case study; OOD flag verified against held-out error; wind-aware emitter exposure; MLflow tracking | Day 32 |
| **Could (Stretch)** | Model B gap reconstruction vs IDW and climatology; 7-day composite & nearest-clear-date alternatives; hosted public live demo; population/built-up ablations | Day 35 |

**Out of scope:** forecasting, ConvLSTM, surface-concentration calibration, a fourth training
city, PostGIS, drift/retrain automation, per-request retraining, a 1 km grid, 1 km downscaling
claims, coastal modelling (water is masked; coastal cells are warned only).

## Phases and gates (schedule: ~38 working days + 4 buffer)

| Phase | Days | Gate |
|---|---|---|
| 0. Feasibility gates | 1–3 | **G0:** one month of Delhi NO₂ in Parquet + feasibility note |
| 1. Delhi data pipeline | 4–8 | **G1:** Delhi complete for chosen window; no missing months |
| 2. Features and baselines | 9–12 | **G2:** baselines scored; block size chosen |
| 3. Model A, uncertainty, validation | 13–18 | **G3:** Model A checkpoint (beats climatology or reported honestly) |
| 4. Live polygon app | 19–28 | **G4 (MVP):** map from a drawn polygon without retraining |
| 5. Multi-city training and transfer | 29–32 | **G5 (Complete):** transfer table and OOD evidence |
| 6. Gap reconstruction, time-boxed | 33–35 | **G6 (Stretch):** Model B beats IDW + climatology on contiguous gaps |
| 7. Polish | 36–38 | README, parity tests, walkthrough |

## Locked Stage-0 decisions

- **D1** Target = retrieved product. No ground-truth accuracy claims.
- **D2** Observed-cell rule: a cell is *observed* when ≥50% (sensitivity 30/70%) of its area is
  covered by valid pixels; several orbits combine as a valid-area-weighted mean; each cell
  stores `n_valid_fraction` and sub-pixel spread.
- **D3** Version tracking: processor/algorithm version stored per image; consistent-version
  training window (start ~mid-2022, confirm from the version plot); version-shift test after
  the November 2025 processor update; no pre-2021 data (known low bias).
- **D4** One weather source, one date limit: ERA5-Land hourly default, BLH from ERA5 single
  levels (confirm in Stage 0); date picker capped at the earliest source limit; weather matched
  to overpass via UTC hour from cell longitude.
- **D5** Quota budget: measured compute cost, daily limit, quota log, single GEE project.
- **D6** Error taxonomy: **observed / reconstructed / insufficient (reason: clouds, haze,
  water, low coverage) / system_error (GEE, OSM, weather, outage)**. `system_error` is **never
  cached as data**. A missing-overpass date is `no_overpass`, not `insufficient`.
- **D7** Demo: local run + pre-computed cached polygons; hosted endpoint only if quota allows.
- **D8** Grid/lags/study area: 0.05° grid; lagged weather (previous-day wind, 24 h/48 h mean
  wind, wind × BLH ventilation proxy, previous-day precipitation) built from weather only;
  Delhi study area = **NCR bounding box** (not NCT alone).

## Repository layout

```
configs/        default.yaml — grid, thresholds, study areas, versions (locked decisions)
src/no2map/     package code (grid, states, config, ...)
scripts/        stage0/ feasibility probes; later phases add pipeline scripts
tests/          unit tests (grid scheme, observed-cell rule, taxonomy)
docs/           feasibility note, data dictionary, retraining note
data/           gitignored: Parquet exports, reports, cache
```

## Quick start (Phase 0)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[gee,dev]"
```

Then register the GEE project (Stage 0 step 2) and open the auth dialogue for Earth Engine:

```powershell
python scripts\ee_auth.py
```

Run the Stage-0 probes (each writes a report under `data\stage0\`):

```powershell
python scripts\stage0\no2_one_month_probe.py   # month of Delhi NO2: units, QA, valid fraction
python scripts\stage0\version_drift.py         # monthly counts by processor version (D3)
python scripts\stage0\weather_latency.py       # ERA5-Land bands, BLH source, latest dates (D4)
python scripts\stage0\cell_report.py           # NCR cell count, water cutoffs, coastline (D8)
python scripts\stage0\quota_cost.py            # one-image compute cost (D5)
```

Feasibility findings go into `docs/feasibility.md` — that plus the Parquet month is Gate G0.

## Limits (stated, per plan §7 and the risk register)

- Retrieval bias: aerosols/haze can be misread as cloud; winter-haze gaps are seasonal.
- Reconstructed maps are *modelling*, not observation; evaluated only on real gap shapes.
- Single-day retrievals are noisy — daily, weekly and monthly aggregations all reported.
- Columns are not surface concentrations; coastal cells are excluded with a warning.
- Coverage, model version, data versions and warnings are shown on every output.
