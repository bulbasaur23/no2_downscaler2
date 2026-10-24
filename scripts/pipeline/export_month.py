"""Phase 1 export worker (Gate G1): city-month Parquet files, resumable.

Orchestrates acquisition (NO2), weather and statics into one monthly Parquet
per city, with manifest-driven resume semantics (src/no2map/manifest.py):

    python scripts/pipeline/export_month.py --city delhi_ncr --month 2025-01
    python scripts/pipeline/export_month.py --city delhi_ncr --window auto

The 'window auto' mode reads configs/training.window_* and exclude_months and
exports only months still tagged pending/failed in the manifest, so a run can
be interrupted at any point and resumed without re-spending Earth Engine work.

Every monthly file stores the D6 cell states (observed / insufficient /
no_overpass), the D2 coverage numbers, the D4/D8 weather features, the static
summaries, and the D3 processor versions touched by the month - all in one
place, joined by (cell_id, date).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from no2map.acquisition import (
    cells_featurecollection,
    convert_day_rows,
    day_no2_cell_sums,
    month_days,
    month_processor_versions,
    no2_projection_scale,
)
from no2map.weather import day_weather_stack
from no2map.statics import static_features_per_cell
from no2map.manifest import ExportManifest
from no2map.features import FEATURE_SCHEMA_VERSION  # single parity-pin source

EXPORTS_DIR = ROOT / "data" / "exports"


def export_month(city: str, month: str, cfg: dict, manifest: ExportManifest,
                 no2_scale: float, weather_scale: float,
                 statics_by_city: dict[str, pd.DataFrame],
                 main_threshold: float) -> tuple[str, dict]:
    """Export one city-month. Never raises for empty orbits - empty days become
    no_overpass rows (D6). Returns (filename, stats)."""
    area = cfg["study_areas"][city]
    cells_fc = cells_featurecollection(area)

    no2_rows = []
    weather_rows = []
    t0 = time.perf_counter()
    for date_str in month_days(month):
        sums = day_no2_cell_sums(area, date_str, cells_fc, no2_scale)
        day_rows = convert_day_rows(sums, main_threshold)
        for r in day_rows:
            r["date"] = date_str  # NO2 rows carry the same (cell_id, date) key as weather
        no2_rows.extend(day_rows)
        weather_rows.extend(day_weather_stack(date_str, weather_scale, cells_fc))
    elapsed = round(time.perf_counter() - t0, 1)

    df_no2 = pd.DataFrame(no2_rows)
    df_wx = pd.DataFrame(weather_rows)
    df_static = statics_by_city[city]

    df = (
        df_no2.merge(df_wx, on=["cell_id", "date"], how="left")
        .merge(df_static, on="cell_id", how="left")
    )
    df.insert(0, "city", city)

    if df.empty:
        raise RuntimeError(f"no rows produced for {city} {month} (empty collection?)")

    out_dir = EXPORTS_DIR / city
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{city}_{month}.parquet"
    df.to_parquet(out_path, index=False)

    versions = month_processor_versions(int(month[:4]), int(month[5:7]))
    manifest.record_success(
        city, month, str(out_path.relative_to(ROOT)), int(len(df)),
        FEATURE_SCHEMA_VERSION, versions,
    )
    manifest.save()

    stats = {
        "rows": int(len(df)),
        "states": df["state"].value_counts().to_dict(),
        "n_cells": df["cell_id"].nunique(),
        "processor_versions": versions,
        "seconds": elapsed,
        "path": str(out_path),
    }
    return out_path.name, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 city-month export worker")
    ap.add_argument("--city", required=True, choices=sorted([
        "delhi_ncr", "bengaluru", "kolkata", "hyderabad"
    ]))
    ap.add_argument("--month", help="YYYY-MM (single month)")
    ap.add_argument("--window", choices=["auto"], help="export everything pending from cfg training window")
    args = ap.parse_args()

    if bool(args.month) == (args.window == "auto"):
        ap.error("use exactly one of --month YYYY-MM or --window auto")

    from no2map.eerun import initialize_ee

    cfg = initialize_ee()  # Earth Engine session (D5: one project, one place)
    manifest = ExportManifest.load(EXPORTS_DIR / "manifest.json")

    months_list: list[str]
    if args.month:
        months_list = [args.month]
    else:
        plan = cfg["training"]
        months_list = [
            m for m in ExportManifest.months_between(plan["window_start"][:7], plan["window_end"][:7])
            if m not in set(plan.get("exclude_months", []))
        ]
        # months_todo excludes `skipped` (two-strike exhausted) months - a
        # plain is_done filter would retry them forever (CodeRabbit finding)
        months_list = manifest.months_todo(args.city, months_list)
        if not months_list:
            print(f"[{args.city}] manifest says nothing pending - nothing to do.")
            return

    print("[info] measuring L3 native scale (D2 check) ...")
    no2_scale = no2_projection_scale()
    print(f"[info] L3 nominal scale: {no2_scale} m")

    main_threshold = cfg["observed_cell_rule"]["main"]
    weather_scale = 11132.0  # ~9 km diagnostics scale for weather (native ~9-27 km)

    # Static features: cached per city; identical month to month by design.
    static_cache_file = EXPORTS_DIR / f"static_{args.city}.parquet"
    if static_cache_file.exists():
        statics_by_city = {args.city: pd.read_parquet(static_cache_file)}
        print(f"[info] statics loaded from cache: {static_cache_file}")
    else:
        cells_fc = cells_featurecollection(cfg["study_areas"][args.city])
        statics_by_city = {args.city: pd.DataFrame(static_features_per_cell(cells_fc, cfg))}
        static_cache_file.parent.mkdir(parents=True, exist_ok=True)
        statics_by_city[args.city].to_parquet(static_cache_file, index=False)
        print(f"[info] statics computed for {len(statics_by_city[args.city])} cells")

    for i, month in enumerate(months_list, start=1):
        manifest.ensure_pending(args.city, [month])  # so failures are recordable
        print(f"[{args.city}] month {i}/{len(months_list)}: {month} ...", flush=True)
        try:
            name, stats = export_month(
                args.city, month, cfg, manifest,
                no2_scale, weather_scale, statics_by_city, main_threshold,
            )
            print(f"  -> {name}  rows={stats['rows']}  states={stats['states']}  "
                  f"versions={stats['processor_versions']}  secs={stats['seconds']}")
        except Exception as exc:  # noqa: BLE001 - record, decide, continue
            status = manifest.record_failure(args.city, month, repr(exc))
            manifest.save()
            print(f"  -> FAILED ({status}): {exc!r}")

    pending = [m for m in months_list if not manifest.is_done(args.city, m)]
    print(f"[{args.city}] still todo: {pending or 'none'}")


if __name__ == "__main__":
    main()
