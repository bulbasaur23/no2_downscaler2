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

## Status: gates on real data (Delhi NCR)

| Gate | Status | Evidence |
|---|---|---|
| **G0** feasibility | PASS | `docs/feasibility.md`, Stage-0 reports in `data/stage0/` |
| **G1** Delhi complete | **PASS** | 30 exported months 2023-04 → 2025-09, 0 failures, 0 warnings — `data/exports/g1_report_delhi_ncr.json` |
| **G2** baselines + block size | **PASS (advice)** | residual variogram range **59.5 km**, sill 632.4 → block **89,278 m** with a 30,000 m buffer is feasible; 119 km impractical — `data/exports/g2_variogram_delhi_ncr.json` |
| **G3** Model A checkpoint | **PASS as `honest_report`** | see below — `data/exports/g3_report_delhi_ncr.json` |
| **G4** live polygon app | code complete, **parity run pending** | app + live path built; `scripts/pipeline/parity_check.py` needs Earth Engine auth |

### G3 result, stated plainly

Three seeds, 89 km spatial blocks + 30 km gap, per-cell climatology as the honesty anchor:

| Protocol | Model A RMSE | Climatology RMSE | Delta | Beyond seed noise? |
|---|---|---|---|---|
| Blocked (unseen areas) | 38.5 ± 0.2 | 36.45 | **+2.03 ± 0.20** (worse) | yes, worse |
| Chronological (unseen future dates) | 18.76 ± 0.02 | 20.34 | **−1.58 ± 0.02** (better) | yes, better |

- Anomaly skill vs climatology: **−0.10 (blocked)**, **+0.15 (chronological)**
- Static-only baseline: 41.95 blocked → weather/lags/temporal features are worth 3.68 µmol/m²
- **Verdict: `honest_report`.** Model A tracks day-to-day and seasonal level well and generalises
  forward in time, but it does **not** beat per-cell climatology at guessing a cell it has never
  seen. The plan's fallback is therefore active: the app returns **observed values** where TROPOMI
  has them, and where it does not it shows the Model A estimate **next to its climatology anchor**,
  both explicitly labelled, with the G3 verdict on screen. Nothing is tuned on the held-out future.

What this means for a user: the observed/reconstructed distinction is the load-bearing one. Treat
"observed" cells as measurement and "reconstructed" cells as a model-informed estimate whose
per-cell climatology is shown beside it — not as a measurement you can act on alone.

### Known blockers

- **Earth Engine permission**: the caller currently lacks
  `serviceusage.services.use` on project `alan-no2-downscaler`, so any fresh fetch fails. This
  blocks the G4 real-data parity run and the remaining 2025-11/12 exports needed for the
  D3 version-shift test. Re-authenticate / grant the role, then rerun the two commands above.
- The version-shift test (`scripts/pipeline/version_shift_test.py`) stays unrun until those two
  post-update months exist; it is not tuned against, only reported.

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
src/no2map/     package: grid, states, features (the train/serve contract), baselines,
                model (blocked + chronological runners), conformal, artifact (serving
                bundle), live (polygon request path), cache, service (FastAPI) + web/
scripts/        stage0/ feasibility probes; pipeline/ export, validation, training, parity
tests/          unit tests for every contract above (277 tests)
docs/           feasibility note, data dictionary
data/           gitignored: Parquet exports, runs, reports, cache
```

## Install

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

## Running the pipeline (G1 → G4)

All commands are resumable/idempotent except the export worker, which must be run **one process
per city at a time** (it writes the manifest).

```powershell
# G1: export the configured window for one city (resumable; skips done months)
python scripts\pipeline\export_month.py --city delhi_ncr --window auto

# G1 completeness report (month presence, schema, states, versions)
python scripts\pipeline\validate_g1.py --city delhi_ncr

# G2: residual variogram -> advisory block size and feasibility verdict
python scripts\pipeline\variogram_g2.py --city delhi_ncr

# G3: full evidence — blocked seed sweep + chronological seed sweep + static-only
python scripts\pipeline\validate_g3.py --city delhi_ncr --block-m 89278 --buffer-m 30000

# Train Model A and persist the serving bundle (model + conformal + climatology anchor)
python scripts\pipeline\train_model_a.py --city delhi_ncr --block-m 89278 --buffer-m 30000 --chron

# G4: real-data parity, training path vs live path (needs Earth Engine auth)
python scripts\pipeline\parity_check.py --city delhi_ncr --date 2025-02-10 --cells 12
```

### Running the app

```powershell
.venv\Scripts\python -m uvicorn no2map.service:create_app --factory --port 8000
```

Open <http://127.0.0.1:8000/> — draw a polygon (or press a demo region), pick a date (capped at
"data as of" per D4), and map it. The UI shows the observed/reconstructed split, coverage and
uncertainty layers, a per-cell inspector, the climatology anchor, the G3 validation note, and CSV
export of the exact cells returned. Results are cached per cell on
(cell, date, feature schema, processor version, model version), so re-querying the same date costs
no Earth Engine work. Predictions on dates inside the training window are flagged in-sample.

## Limits (stated, per plan §7 and the risk register)

- Retrieval bias: aerosols/haze can be misread as cloud; winter-haze gaps are seasonal.
- **Model A does not beat per-cell climatology on unseen spatial blocks (G3 `honest_report`)**;
  reconstructed cells are shown with their climatology anchor and are not measurement.
- Reconstructed maps are *modelling*, not observation; evaluated only on real gap shapes.
- Single-day retrievals are noisy — daily, weekly and monthly aggregations all reported.
- Columns are not surface concentrations; coastal cells are excluded with a warning.
- Coverage, model version, data versions and warnings are shown on every output.
