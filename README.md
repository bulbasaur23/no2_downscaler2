# AI-Based NO₂ Mapping Platform

Mapping of the **TROPOMI-retrieved tropospheric NO₂ column (µmol/m²)** over Indian cities on a
**0.05° grid**, with a live *draw-any-polygon* query app, gap context, calibrated uncertainty,
and **honest validation**.

> **What this is NOT:** this is not ground-truth NO₂. The target is the satellite-retrieved
> product (GEE `COPERNICUS/S5P/OFFL/L3_NO2`). Accuracy is reported **only against TROPOMI**.
> Station data (CPCB, optional) gives correlation and bias only — never accuracy (D1).

## Concept

1. **Fixed 0.05° grid** — TROPOMI's native pixel is ~5.5 × 3.5 km, so a finer grid would repeat
   one retrieval across many cells and inflate the apparent sample size (D8). Static layers are
   sampled at ~1 km and *summarised per cell*; the shipped schema has **elevation and water only**
   (roads/emitters are a planned Should-tier addition, not yet built — see
   `docs/data_dictionary.md`).
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
| **G4** live polygon app | **PASS** | real-data parity 12/12 cells, **max feature/prediction diff 0.0** (`data/exports/g4_parity_delhi_ncr.json`); a real HTTP job on 2025-11-25 returned observed cells + climatology anchor + G3 note, and CSV export returned 200 with the state column |
| **D3** version-shift test | **run, honest result** | Model A still beats climatology in both post-update months, but its bias swings +10.6 → −11.3 µmol/m² — see below; `data/shift/delhi_ncr_version_shift.json` |
| **G5** multi-city transfer | **in progress** | Bengaluru export running; Kolkata/Hyderabad exports queued; `scripts/pipeline/transfer_g5.py` implements leave-one-city-out + OOD evidence and refuses a partial training window |

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

### Version-shift result (post-2025 processor update), stated plainly

The Nov-2025 processor update did not land on a month boundary. Measured per day over **Delhi's
overpass window** (not the global collection): 2025-11 days 1–3 had no overpass, days 4–21 were
still `2.8.0`, day 22 was mixed, and days 23–30 were `02.09.01`; 2025-12 was pure `02.09.01`.
The eval set therefore uses **8 days of 2025-11 + all 31 days of 2025-12**, every one version-pure
and outside the training versions (`2.5.0`–`2.8.0`). The 19 old-version days and 3 empty days are
recorded in the report with their reason — excluded on the treatment, never on outcomes. Nothing
was refit on the eval days.

| Month | Eval days | Model A RMSE | Climatology RMSE | Model A bias | Level r (A vs clim) | Pattern r (A vs clim) | Interval coverage |
|---|---|---|---|---|---|---|---|
| 2025-11 | 8 | **41.93** | 42.60 | **+10.55** | 0.56 vs −0.07 | 0.62 vs 0.62 | 0.956 |
| 2025-12 | 31 | **52.60** | 55.77 | **−11.31** | 0.23 vs 0.09 | 0.60 vs 0.63 | 0.911 |

Reading it honestly: Model A **still beats per-cell climatology on cellwise RMSE in both months**,
and it is the only method that tracks the day-to-day level in November (r 0.56 vs −0.07), so the
update did not break the model. But the bias **flips sign between the two months** (+10.6 then
−11.3 µmol/m²) while the frozen intervals stay over-covered (0.91–0.96 at a nominal 0.90) and very
wide (72.5 µmol/m²). That is a real distribution shift, not seed noise — the honest reading is
*usable ranking/level skill with an unstable absolute level across the processor update*. Every
output therefore carries the processor versions, and reconstructed values should not be compared
across the update without saying so.

Caveat on precision: the 2025-11 column rests on **8 days only**, so its level correlation (8
points) and bias are indicative rather than precise; 2025-12 (31 days) is the stronger half of the
evidence.

### Known blockers

- ~~**Earth Engine permission**~~ **resolved 2026-10-26**: the caller had lacked
  `serviceusage.services.use` on project `alan-no2-downscaler`. After re-auth the 2025-11 month was
  re-exported (48,300 rows; 3 days genuinely had no overpass and are recorded as `no_overpass` —
  previously an empty-day bug crashed the whole month, see git history).
- **Road/emitter static features are not built** (scope gap, see Limits): the model uses elevation
  and water only, so the Phase-5 emitter-count check cannot run.
- **The Nov-2025 processor update did not land on a month boundary.** Measured over Delhi's
  overpass window: 2025-11 days 1–3 have no overpass, days 4–21 are still `2.8.0` (a training
  version), day 22 is mixed, days 23–30 are `02.09.01`; 2025-12 is pure `02.09.01`. The shift runner
  therefore keeps only version-pure eval days and records every excluded day with its reason —
  exclusion is on the treatment (processor), never on outcomes.

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
                bundle), live (polygon request path), cache, service (FastAPI) + web/,
                transfer (leave-one-city-out + OOD)
scripts/        stage0/ feasibility probes; pipeline/ export, validation, training, parity,
                version-shift, transfer
tests/          unit tests for every contract above (308 tests)
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

## Running the pipeline (G1 → G5)

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

# D3: frozen version-shift test on version-pure post-update days
python scripts\pipeline\version_shift_test.py --city delhi_ncr --block-m 89278 --buffer-m 30000

# G5: leave-one-city-out transfer + OOD evidence (needs the training cities exported)
python scripts\pipeline\transfer_g5.py --cities delhi_ncr,bengaluru,kolkata
# G5 case study: train on the three cities, score the untouched city
python scripts\pipeline\transfer_g5.py --cities delhi_ncr,bengaluru,kolkata,hyderabad --test-cities hyderabad
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
- **Road-density and emitter features are not implemented** (the plan's Phase 2 lists them);
  the model uses elevation + water as static context. The Phase-5 "emitter count per city" check
  therefore cannot run yet, and transfer results reflect the smaller feature set.
- Coverage, model version, data versions and warnings are shown on every output.
