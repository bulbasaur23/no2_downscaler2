"""G4 parity check on REAL data: training rows vs the live request path.

Plan (Phase 4): "Parity test: same polygon and date through training and live
paths gives identical features, including clipping, lags and BLH."

This script takes a grid-aligned polygon and a date that already exist in an
exported month, then:
  1. runs the live path (EE NO2 + weather/lags + cached statics + shared
     `build_features`),
  2. loads the same cells/date from the training parquet,
  3. compares D6 states, every weather/lag column, and the 24 model features
     cell by cell, plus the artifact's predictions.

Differences only appear if the live fetch/join/mapping diverges from export,
or if GEE reprocessed the source since the export (reported separately, since
that is a data revision, not a code defect).

Usage:
  python scripts/pipeline/parity_check.py --city delhi_ncr --date 2025-02-10 \
      [--cells 12] [--half-deg 0.15]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from no2map.acquisition import no2_projection_scale          # noqa: E402
from no2map.eerun import initialize_ee                        # noqa: E402
from no2map.features import (FEATURE_GROUPS, TARGET,          # noqa: E402
                             build_features, feature_block)
from no2map.grid import cells_in_bbox                         # noqa: E402
from no2map.live import (LivePredictor, load_latest_artifact,  # noqa: E402
                         load_statics)
from no2map.weather import WEATHER_COLUMNS                    # noqa: E402

EXPORTS = ROOT / "data" / "exports"
RUNS = ROOT / "data" / "runs"


def _max_abs_diff(a: pd.Series, b: pd.Series) -> float | None:
    x = a.to_numpy(dtype=float)
    y = b.to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return 0.0 if (np.isfinite(x) == np.isfinite(y)).all() else None
    d = float(np.max(np.abs(x[m] - y[m])))
    # any NaN pattern difference is a real mismatch, not a small number
    if not (np.isfinite(x) == np.isfinite(y)).all():
        return float("inf")
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", default="delhi_ncr")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD inside an exported month")
    ap.add_argument("--cells", type=int, default=12)
    ap.add_argument("--half-deg", type=float, default=0.15,
                    help="half-width of the probe box around the city centre")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="numeric tolerance for feature/weather equality")
    args = ap.parse_args()

    month = args.date[:7]
    parquet = EXPORTS / args.city / f"{args.city}_{month}.parquet"
    if not parquet.exists():
        print(f"[parity] missing export: {parquet}")
        return 2

    cfg = initialize_ee()
    area = cfg["study_areas"][args.city]
    lat0 = (area["lat_min"] + area["lat_max"]) / 2.0
    lon0 = (area["lon_min"] + area["lon_max"]) / 2.0
    cells = list(cells_in_bbox(lat0 - args.half_deg, lat0 + args.half_deg,
                               lon0 - args.half_deg, lon0 + args.half_deg))[:args.cells]
    if not cells:
        print("[parity] no cells in probe box")
        return 2
    lon_min = min(c.lon_min for c in cells)
    lon_max = max(c.lon_max for c in cells)
    lat_min = min(c.lat_min for c in cells)
    lat_max = max(c.lat_max for c in cells)
    polygon = [(lon_min, lat_min), (lon_max, lat_min),
               (lon_max, lat_max), (lon_min, lat_max)]
    cell_ids = sorted(c.id for c in cells)

    print(f"[parity] {args.city} {args.date}: {len(cell_ids)} cells "
          f"({lon_min:.2f}..{lon_max:.2f}, {lat_min:.2f}..{lat_max:.2f})")
    print(f"[parity] NO2 native scale: {no2_projection_scale()} m")

    art, run_name = load_latest_artifact(RUNS, args.city)
    predictor = LivePredictor(
        city=args.city, city_cfg=area, artifact=art,
        statics=load_statics(EXPORTS, args.city), run_name=run_name,
        main_threshold=float(cfg["observed_cell_rule"]["main"]))
    live = predictor.assemble(args.date, cell_ids).sort_values("cell_id")
    live = live.reset_index(drop=True)

    train = pd.read_parquet(parquet)
    train = train[(train["date"] == args.date) &
                  (train["cell_id"].isin(cell_ids))].sort_values("cell_id")
    train = train.reset_index(drop=True)
    if len(train) != len(live):
        print(f"[parity] row mismatch: live {len(live)} vs export {len(train)}")
        return 2

    out: dict = {
        "city": args.city, "date": args.date, "cells": cell_ids,
        "model_run": run_name, "tolerance": args.tol,
        "states": {}, "numerics": {}, "features": {},
        "prediction_max_abs_diff": None,
    }

    # D6 state agreement
    agree = int((live["state"].to_numpy() == train["state"].to_numpy()).sum())
    out["states"] = {"agree": agree, "n": int(len(live)),
                     "live": live["state"].value_counts().to_dict(),
                     "export": train["state"].value_counts().to_dict()}

    # NO2 target (only where both observed) - reprocessing shows up here first
    obs = (live["state"] == "observed") & (train["state"] == "observed")
    out["numerics"]["value_umol_m2"] = (
        _max_abs_diff(live.loc[obs, TARGET], train.loc[obs, TARGET])
        if obs.any() else None)

    # weather + lags: every column the export stores
    for c in WEATHER_COLUMNS:
        if c in train.columns and c in live.columns:
            out["numerics"][c] = _max_abs_diff(live[c], train[c])

    # features: the 24 the model actually consumes
    live_feat = build_features(live)
    train_feat = build_features(train)
    for c in feature_block(live_feat, list(FEATURE_GROUPS)):
        out["features"][c] = _max_abs_diff(live_feat[c], train_feat[c])

    # predictions through the SAME artifact
    p_live = art.predict(live)
    p_train = art.predict(train)
    pred_diff = _max_abs_diff(p_live["pred"], p_train["pred"])
    out["prediction_max_abs_diff"] = pred_diff

    numeric_bad = {k: v for k, v in {**out["numerics"], **out["features"]}.items()
                   if v is not None and v > args.tol}
    state_ok = agree == len(live)
    pred_ok = pred_diff is not None and pred_diff <= max(args.tol, 1e-9)
    value_revised = (out["numerics"]["value_umol_m2"] is not None
                     and out["numerics"]["value_umol_m2"] > args.tol)

    out["verdict"] = {
        "states_match": state_ok,
        "numerics_within_tolerance": not numeric_bad,
        "predictions_match": pred_ok,
        "no2_source_revised_since_export": value_revised,
        "bad_columns": numeric_bad,
        "pass": bool(state_ok and not numeric_bad and pred_ok),
    }

    dst = EXPORTS / f"g4_parity_{args.city}.json"
    dst.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    worst = max((v for v in out["features"].values() if v is not None),
                default=0.0)
    print(f"[parity] states {agree}/{len(live)} agree | "
          f"features max |diff| = {worst:.3e} | "
          f"prediction max |diff| = {pred_diff if pred_diff is None else f'{pred_diff:.3e}'}")
    if value_revised:
        print(f"[parity] NOTE: observed NO2 differs from the export by "
              f"{out['numerics']['value_umol_m2']:.4f} umol/m2 - GEE source was "
              f"reprocessed after the export (not a code defect)")
    if numeric_bad:
        top = sorted(numeric_bad.items(), key=lambda kv: kv[1], reverse=True)[:8]
        print(f"[parity] FAIL columns beyond {args.tol:g}: {top}")
    print(f"[parity] verdict: {'PASS' if out['verdict']['pass'] else 'FAIL'}")
    print(f"[parity] report: {dst}")
    return 0 if out["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
